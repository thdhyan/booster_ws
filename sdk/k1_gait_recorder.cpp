// Record the Booster K1 factory walker's gait for use as a sim reference motion.
//
// Why: our PPO velocity policy shuffles because velocity tracking does not
// constrain *how* the robot steps.  A human-like gait needs reference motion
// tracking, and the best reference available is the walk Booster ships on the
// robot's own firmware -- the controller that runs on 700+ real units.
//
// What this captures per step:
//   rt/low_state       22x dq (joint pos), ddq (joint vel), tau_est (JOINT EFFORT),
//                      plus motor temperature and mode
//   rt/odometer_state  x, y, theta  -> integrated into a world trajectory
//   IMU                rpy, gyro, acc  (from LowState)
//
// The speed sweep is the other point: it measures the factory walker's ACTUAL
// achievable speed, which tells us whether the 3 m/s target is even on the
// hardware before we spend GPU-weeks chasing it.
//
// SAFETY: this commands a real biped.  It only uses the supported high-level
// B1LocoClient.Move API, ramps rather than steps, and returns to a stand at
// the end of every phase and on SIGINT.  Keep a physical e-stop within reach
// and clear the floor.  Speed caps are conservative by default.
//
// Build:
//   g++ -std=c++17 k1_gait_recorder.cpp -o k1_gait_recorder \
//       -I sdk/booster_robotics_sdk/include -I sdk/booster_robotics_sdk/include/booster/third_party
// Run:
//   ./k1_gait_recorder 192.168.1.100 192.168.1.100 --speeds 0,0.2,0.4,0.6,0.8,1.0 --hold 6
//
// The two args are the local and robot network interface IPs (DDS needs both).

#include <booster/idl/b1/FallDownState.h>
#include <booster/idl/b1/LowState.h>
#include <booster/idl/b1/Odometer.h>
#include <booster/idl/b1/RobotStates.h>
#include <booster/idl/b1/RobocupBehaviorStatus.h>
#include <booster/robot/b1/b1_api_const.hpp>
#include <booster/robot/b1/b1_loco_client.hpp>
#include <booster/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace booster::robot;
using namespace booster::common;
using namespace booster_interface::msg;

namespace {

constexpr int    kLegJoints   = 12;   // 6 per leg: 3 hip, 1 knee, 2 ankle
constexpr double kCycleDt     = 1.0 / 50.0;  // LowState is 500 Hz; we log at 50 Hz
                                 // to match the booster_train motion CSV contract
std::atomic<bool> g_stop{false};
std::mutex        g_mtx;

struct Sample {
    double  t = 0.0;
    double  cmd_vx = 0.0, cmd_vy = 0.0, cmd_vyaw = 0.0;
    double  phase_vx = 0.0;                 // commanded speed for this phase
    std::vector<double> dq, ddq, tau;       // K1 joint order
    double  rpy[3] = {0, 0, 0}, gyro[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
    double  od_x = 0.0, od_y = 0.0, od_theta = 0.0;
    bool    has_odom = false;
    // Action labels and safety metadata.
    int32_t current_mode = 0;              // kDamping / kPrepare / kWalking
    int32_t current_body_control = 0;
    int32_t n_actions = 0;                 // size of RobotStates.current_actions
    uint64_t action_mask = 0;              // packed current_actions bits
    int32_t fall_down = -1;                // -1 unknown, else state
    int32_t recovery_available = -1;
    int32_t robocup_status = -1;           // soccer behaviour the robot is in
};

Sample g_latest;                 // guarded by g_mtx
std::atomic<double> g_cmd_vx{0.0}, g_cmd_vy{0.0}, g_cmd_vyaw{0.0}, g_phase_vx{0.0};
std::chrono::steady_clock::time_point g_t0;
double g_phase_start = 0.0;

void OnSignal(int) { g_stop = true; }

// --- DDS handlers -----------------------------------------------------------
void LowStateHandler(const void *msg) {
    const auto *ls = static_cast<const LowState *>(msg);
    std::lock_guard<std::mutex> lk(g_mtx);
    g_latest.dq.clear();
    g_latest.ddq.clear();
    g_latest.tau.clear();
    for (const auto &m : ls->motor_state_serial()) {
        g_latest.dq.push_back(m.dq());
        g_latest.ddq.push_back(m.ddq());
        g_latest.tau.push_back(m.tau_est());
    }
    for (const auto &m : ls->motor_state_parallel()) {
        g_latest.dq.push_back(m.dq());
        g_latest.ddq.push_back(m.ddq());
        g_latest.tau.push_back(m.tau_est());
    }
    const auto &imu = ls->imu_state();
    for (int i = 0; i < 3; ++i) {
        g_latest.rpy[i]   = imu.rpy()[i];
        g_latest.gyro[i]  = imu.gyro()[i];
        g_latest.acc[i]   = imu.acc()[i];
    }
}

void OdometerHandler(const void *msg) {
    const auto *od = static_cast<const Odometer *>(msg);
    std::lock_guard<std::mutex> lk(g_mtx);
    g_latest.od_x     = od->x();
    g_latest.od_y     = od->y();
    g_latest.od_theta = od->theta();
    g_latest.has_odom = true;
}

// RobotStates: which mode the robot is in and the active behaviour mask.
void RobotStatesHandler(const void *msg) {
    const auto *rs = static_cast<const booster_interface::msg::RobotStatesMsg *>(msg);
    std::lock_guard<std::mutex> lk(g_mtx);
    g_latest.current_mode = static_cast<int32_t>(rs->current_mode());
    g_latest.current_body_control = static_cast<int32_t>(rs->current_body_control());
    const auto &acts = rs->current_actions();
    g_latest.n_actions = static_cast<int32_t>(acts.size());
    uint64_t mask = 0;
    for (size_t i = 0; i < acts.size() && i < 64; ++i) {
        if (acts[i]) mask |= (uint64_t(1) << i);
    }
    g_latest.action_mask = mask;
}

void FallDownHandler(const void *msg) {
    const auto *fd = static_cast<const FallDownState *>(msg);
    std::lock_guard<std::mutex> lk(g_mtx);
    g_latest.fall_down = static_cast<int32_t>(fd->fall_down_state());
    g_latest.recovery_available = static_cast<int32_t>(fd->is_recovery_available());
}

void RobocupStatusHandler(const void *msg) {
    const auto *rb = static_cast<const booster_interface::msg::RobocupBehaviorStatus *>(msg);
    std::lock_guard<std::mutex> lk(g_mtx);
    g_latest.robocup_status = static_cast<int32_t>(rb->status());
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
}

std::vector<double> ParseList(const std::string &s) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) v.push_back(std::stod(tok));
    }
    return v;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <local_ip> <robot_ip> [--speeds a,b,c] [--hold SEC]"
                     " [--settle SEC] [--out FILE] [--lateral] [--yaw]\n";
        return 2;
    }
    std::vector<double> speeds{0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    double hold = 6.0, settle = 2.5;
    std::string out = "k1_factory_gait.csv";
    bool lateral = false, yaw = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--speeds" && i + 1 < argc)        speeds = ParseList(argv[++i]);
        else if (a == "--hold" && i + 1 < argc)      hold = std::stod(argv[++i]);
        else if (a == "--settle" && i + 1 < argc)    settle = std::stod(argv[++i]);
        else if (a == "--out" && i + 1 < argc)       out = argv[++i];
        else if (a == "--lateral")                  lateral = true;
        else if (a == "--yaw")                      yaw = true;
    }

    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);

    g_t0 = std::chrono::steady_clock::now();
    ChannelFactory::Instance()->Init(0, argv[1]);
    ChannelSubscriber<LowState>          low(booster::robot::b1::kTopicLowState, LowStateHandler);
    ChannelSubscriber<Odometer>          odom(booster::robot::b1::kTopicOdometerState, OdometerHandler);
    ChannelSubscriber<booster_interface::msg::RobotStatesMsg> rstate(booster::robot::b1::kTopicRobotStates, RobotStatesHandler);
    ChannelSubscriber<FallDownState>     fall(booster::robot::b1::kTopicFallDown, FallDownHandler);
    ChannelSubscriber<booster_interface::msg::RobocupBehaviorStatus> robocup(
        booster::robot::b1::kTopicRobocupBehaviorStatus, RobocupStatusHandler);
    low.InitChannel();
    odom.InitChannel();
    rstate.InitChannel();
    fall.InitChannel();
    robocup.InitChannel();

    booster::robot::b1::B1LocoClient loco;
    std::cout << "[rec] changing to kWalking...\n";
    if (loco.ChangeMode(RobotMode::kWalking) != 0) {
        std::cerr << "[rec] FAILED to enter kWalking -- aborting (robot may be "
                     "unreachable at " << argv[2] << ")\n";
        return 3;
    }
    loco.Move(0.0f, 0.0f, 0.0f);

    std::ofstream f(out);
    if (!f) { std::cerr << "[rec] cannot write " << out << "\n"; return 4; }
    f << std::setprecision(9);
    f << "t,cmd_vx,cmd_vy,cmd_vyaw,phase_vx,"
         "od_x,od_y,od_theta,rpy0,rpy1,rpy2,gyro0,gyro1,gyro2,acc0,acc1,acc2"
         // action labels + safety metadata: the (obs -> action) pairs that
         // behaviour cloning needs, plus fall/ behaviour labels.
         ",current_mode,current_body_control,action_mask,n_actions"
         ",fall_down,recovery_available,robocup_status";
    for (int i = 0; i < kLegJoints; ++i) f << ",dq" << i;
    for (int i = 0; i < kLegJoints; ++i) f << ",ddq" << i;
    for (int i = 0; i < kLegJoints; ++i) f << ",tau" << i;
    f << "\n";
    std::cout << "[rec] writing " << out << "\n[rec] hold=" << hold
              << "s settle=" << settle << "s per speed\n";

    auto tick = [&]() {
        while (!g_stop) {
            std::this_thread::sleep_until(
                std::chrono::steady_clock::now() +
                std::chrono::duration<double>(kCycleDt));
            const double t = Now();
            Sample s;
            {
                std::lock_guard<std::mutex> lk(g_mtx);
                s = g_latest;
            }
            s.t = t;
            s.cmd_vx = g_cmd_vx.load();
            s.cmd_vy = g_cmd_vy.load();
            s.cmd_vyaw = g_cmd_vyaw.load();
            s.phase_vx = g_phase_vx.load();
            if (s.dq.size() < static_cast<size_t>(kLegJoints)) continue;
            f << s.t << "," << s.cmd_vx << "," << s.cmd_vy << "," << s.cmd_vyaw
              << "," << s.phase_vx << "," << s.od_x << "," << s.od_y << "," << s.od_theta;
            for (int i = 0; i < 3; ++i) f << "," << s.rpy[i];
            for (int i = 0; i < 3; ++i) f << "," << s.gyro[i];
            for (int i = 0; i < 3; ++i) f << "," << s.acc[i];
            f << "," << s.current_mode << "," << s.current_body_control
              << "," << s.action_mask << "," << s.n_actions
              << "," << s.fall_down << "," << s.recovery_available
              << "," << s.robocup_status;
            for (int i = 0; i < kLegJoints; ++i) f << "," << s.dq[i];
            for (int i = 0; i < kLegJoints; ++i) f << "," << s.ddq[i];
            for (int i = 0; i < kLegJoints; ++i) f << "," << s.tau[i];
            f << "\n";

            // 20 Hz command resend, mirroring the reference repo's live sender.
            if (std::fmod(t, 0.05) < kCycleDt) {
                loco.MoveCommand(static_cast<float>(s.cmd_vx),
                                 static_cast<float>(s.cmd_vy),
                                 static_cast<float>(s.cmd_vyaw));
            }
        }
    };
    std::thread logger(tick);

    for (double v : speeds) {
        if (g_stop) break;
        // Ramp: never step straight to the target.
        const double vx = lateral ? 0.0f : v, vy = lateral ? v : 0.0f;
        g_phase_vx = v;
        std::cout << "[rec] ramp to vx=" << vx << " vy=" << vy << "\n";
        for (int s = 1; s <= 10; ++s) {
            if (g_stop) break;
            g_cmd_vx = vx * s / 10.0;
            g_cmd_vy = vy * s / 10.0;
            loco.MoveCommand(static_cast<float>(g_cmd_vx), static_cast<float>(g_cmd_vy), 0.0f);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        g_phase_start = Now();
        while (!g_stop && Now() - g_phase_start < settle) std::this_thread::sleep_for(std::chrono::milliseconds(20));

        if (yaw) { g_cmd_vx = 0.0; g_cmd_vy = 0.0; g_cmd_vyaw = v; }
        else     { g_cmd_vx = vx; g_cmd_vy = vy; g_cmd_vyaw = 0.0; }
        std::cout << "[rec] hold " << hold << "s at vx=" << g_cmd_vx
                  << " vy=" << g_cmd_vy << " vyaw=" << g_cmd_vyaw << "\n";
        g_phase_start = Now();
        while (!g_stop && Now() - g_phase_start < hold) std::this_thread::sleep_for(std::chrono::milliseconds(20));

        g_cmd_vx = g_cmd_vy = g_cmd_vyaw = 0.0;
        loco.MoveCommand(0.0f, 0.0f, 0.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
    }

    g_stop = true;
    if (logger.joinable()) logger.join();
    f.flush();
    f.close();
    loco.Move(0.0f, 0.0f, 0.0f);
    loco.ChangeMode(RobotMode::kPrepare);   // back to standing; never leave it damping
    std::cout << "[rec] done -> " << out << " (returned to kPrepare)\n";
    return 0;
}
