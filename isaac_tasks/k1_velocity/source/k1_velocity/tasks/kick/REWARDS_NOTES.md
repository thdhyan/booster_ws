## Notes (2026-10-11)

- The goal is fixed at `GOAL_POS = (FIELD_L/2, 0, GOAL_H/2)` (`kick/mdp.py`); `curriculum = None`. Ball resets
  come from `reset_ball_omnireset`: at_ball_shoot 50 %, stand_ready 30 %, walk_up 20 %.
- Final run: 3399 iterations, about 53 % goals.
- Proposed: goal-position and ball-distance curriculum (random goal y and distance, widening on success rate).
