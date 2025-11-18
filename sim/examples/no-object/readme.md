```mermaid
flowchart TD
  IDLE[IDLE]
  CF[CF_ACTION]
  BACK[BACKWARD]
  CHG[CHANGE_DIRECTION]
  FIND[FIND_GOAL]
  FOLLOW[FOLLOW_TRAJECTORY]
  EXIT[EXIT]
  LAND[LAND]
  END[END]

  %% Main transitions from IDLE
  IDLE -->|depth1 and depth2 within MIN to MAX, set CF_action_counter and backward_action_counter| CF
  IDLE -->|otherwise: forward control| IDLE

  %% CF_ACTION transitions
  CF -->|residuals > 0.08 and not last_traj| FIND
  CF -->|CF_action_counter <= 0| BACK
  CF -->|outside True| BACK

  %% FIND_GOAL transitions
  FIND -->|goal_counter <= 0| IDLE

  %% BACKWARD transitions
  BACK -->|backward_action_counter <=0 and dir_changes_completed < 3| CHG
  BACK -->|backward_action_counter <=0 and dir_changes_completed >=3 and finish_CF True| BACK
  BACK -->|outside True| BOUNDARY_CHECK[(handle_boundary_violation / GPIS)]
  BACK -->|...| FOLLOW

  %% Boundary check / GPIS node
  BOUNDARY_CHECK -->|uncertainty < thresh 6.5%| PLAN_TO_END[Plan traj -> send -> last_trajectory=True]
  BOUNDARY_CHECK -->|uncertainty >= thresh| REVERSE_OR_TURN[Reverse previous traj or set target_yaw]

  PLAN_TO_END --> EXIT
  EXIT --> FOLLOW

  REVERSE_OR_TURN -->|has previous trajectory| FOLLOW
  REVERSE_OR_TURN -->|no previous trajectory| CHG

  %% FOLLOW_TRAJECTORY behavior
  FOLLOW -->|at waypoint: inc traj_index| FOLLOW
  FOLLOW -->|if back and arrived last waypoint| IDLE
  FOLLOW -->|if outside and arrived last waypoint| LAND
  FOLLOW -->|if not back and depth triggers CF| CF
  FOLLOW -->|if finished and not back| IDLE

  %% LAND transitions
  EXIT -->|follow traj end -> LAND| LAND
  LAND -->|save data, mission complete| END

  %% Node styles
  style BOUNDARY_CHECK fill:#f9f,stroke:#333,stroke-width:1px
  style PLAN_TO_END fill:#afa,stroke:#333
  style REVERSE_OR_TURN fill:#ffa,stroke:#333


