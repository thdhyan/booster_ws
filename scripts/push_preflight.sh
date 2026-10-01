#!/bin/bash
# =============================================================================
# Blocker-5 preflight for the P6 push chain (zz-bw or any clone).
#
# Catches, BEFORE a launch burns hours on a broken tree:
#   1. booster_train_ref submodule CONTENT dirt - invisible in the PARENT
#      git status (the 2026-09-27 contact-sensor regression was exactly this:
#      parent clean, submodule files reverted)
#   2. gitlink mismatch (submodule HEAD != the sha recorded by the parent)
#   3. the _spawn_k1_urdf contact-API wrapper present in booster.py
#   4. v3 push code present (goal_yaw_curriculum marker) - the zzbw chain
#      requires the rsynced v3 files, not the tree's committed v2
#   5. disk free >= PRE_MIN_FREE_GB (default 60; the FS runs near full)
#   6. run_unbuffered.py at the clone root (K1_TRAIN_SCRIPT used by zzbw
#      launches; downgradable to WARN with PRE_REQUIRE_UNBUFFERED=0)
#   7. frozen base export resolvable (WARN by default - the chain exports it
#      before training; K1_PRE_REQUIRE_BASE=1 makes it fatal for stage hosts)
#
# Usage: push_preflight.sh [repo]     (default: parent dir of this script)
# Gate:  grep '^PREFLIGHT_RESULT=PASS'  (exit code mirrors: 0 pass / 1 fail)
# Env:   PRE_MIN_FREE_GB=60  PRE_REQUIRE_UNBUFFERED=1  K1_PRE_REQUIRE_BASE=0
#        PUSH_BASE_POLICY=models/k1_push_base.pt
# =============================================================================
set -u

REPO="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PRE_MIN_FREE_GB="${PRE_MIN_FREE_GB:-60}"
PRE_REQUIRE_UNBUFFERED="${PRE_REQUIRE_UNBUFFERED:-1}"
K1_PRE_REQUIRE_BASE="${K1_PRE_REQUIRE_BASE:-0}"
FAIL=0

chk() { # chk <NAME> <PASS|FAIL|WARN> <detail>
    printf 'PREFLIGHT_%s=%s  # %s\n' "$1" "$2" "$3"
    [ "$2" = "FAIL" ] && FAIL=1
    return 0
}

if [ ! -d "$REPO/.git" ] && [ ! -f "$REPO/.git" ]; then
    echo "PREFLIGHT_RESULT=FAIL  # not a git repo: $REPO"
    exit 1
fi
SUB="$REPO/isaac_tasks/booster_train_ref"
echo "[preflight] repo=$REPO"

# ---- 1. submodule content dirt (the historical silent killer) --------------
if [ ! -d "$SUB" ]; then
    chk SUBMODULE_CONTENT FAIL "submodule not checked out: $SUB"
else
    DIRT=$(git -C "$SUB" status --porcelain 2>/dev/null | head -5)
    if [ -n "$DIRT" ]; then
        chk SUBMODULE_CONTENT FAIL "dirt in booster_train_ref: $(echo "$DIRT" | tr '\n' ' ')"
    else
        chk SUBMODULE_CONTENT PASS "clean"
    fi
fi

# ---- 2. gitlink == submodule HEAD -----------------------------------------
REC=$(git -C "$REPO" ls-tree HEAD isaac_tasks/booster_train_ref 2>/dev/null | awk '{print $3}')
ACT=$(git -C "$SUB" rev-parse HEAD 2>/dev/null || true)
if [ -z "$REC" ] || [ -z "$ACT" ]; then
    chk GITLINK FAIL "recorded='${REC:-?}' actual='${ACT:-?}'"
elif [ "$REC" = "$ACT" ]; then
    chk GITLINK PASS "${REC:0:12}"
else
    chk GITLINK FAIL "recorded=${REC:0:12} actual=${ACT:0:12} (submodule moved/behind)"
fi

# ---- 3. contact-API wrapper (root-cause fix bf3342e) -----------------------
BPY="$SUB/source/booster_train/booster_train/assets/robots/booster.py"
if [ ! -f "$BPY" ]; then
    chk CONTACT_FN FAIL "missing $BPY"
elif grep -q "_spawn_k1_urdf" "$BPY"; then
    chk CONTACT_FN PASS "_spawn_k1_urdf in $(basename "$BPY")"
else
    chk CONTACT_FN FAIL "_spawn_k1_urdf ABSENT from $BPY (contact sensors will regress)"
fi

# ---- 4. v3 push code present ----------------------------------------------
V3="$REPO/isaac_tasks/k1_velocity/source/k1_velocity/tasks/push/push_mdp.py"
if [ -f "$V3" ] && grep -q "goal_yaw_curriculum" "$V3"; then
    chk PUSH_V3_CODE PASS "goal_yaw_curriculum marker found"
else
    chk PUSH_V3_CODE FAIL "v3 marker absent - rsync the v3 push files first: $V3"
fi

# ---- 5. disk free ----------------------------------------------------------
AVAIL=$(df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL" ] && [ "$AVAIL" -ge "$PRE_MIN_FREE_GB" ]; then
    chk DISK PASS "${AVAIL}G free (>= ${PRE_MIN_FREE_GB}G)"
else
    chk DISK FAIL "${AVAIL:-?}G free (< ${PRE_MIN_FREE_GB}G)"
fi

# ---- 6. run_unbuffered.py --------------------------------------------------
if [ -f "$REPO/run_unbuffered.py" ]; then
    chk RUN_UNBUFFERED PASS "present"
elif [ "$PRE_REQUIRE_UNBUFFERED" = "1" ]; then
    chk RUN_UNBUFFERED FAIL "missing $REPO/run_unbuffered.py"
else
    chk RUN_UNBUFFERED WARN "missing (not required in this mode)"
fi

# ---- 7. frozen base export -------------------------------------------------
BASE="${PUSH_BASE_POLICY:-models/k1_push_base.pt}"
case "$BASE" in /*) BASE_PATH="$BASE" ;; *) BASE_PATH="$REPO/$BASE" ;; esac
if [ -f "$BASE_PATH" ]; then
    chk FROZEN_BASE PASS "$BASE_PATH"
elif [ "$K1_PRE_REQUIRE_BASE" = "1" ]; then
    chk FROZEN_BASE FAIL "$BASE_PATH missing (export first)"
else
    chk FROZEN_BASE WARN "$BASE_PATH not yet exported (chain exports before launch)"
fi

# ---- result ----------------------------------------------------------------
if [ "$FAIL" -eq 0 ]; then
    echo "PREFLIGHT_RESULT=PASS"
    exit 0
else
    echo "PREFLIGHT_RESULT=FAIL"
    exit 1
fi
