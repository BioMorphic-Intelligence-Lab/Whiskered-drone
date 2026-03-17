/**
 * Whisker-based tactile flight controller for Crazyflie
 *
 * Copyright (C) 2026 Chaoxiang Ye
 *
 * This file implements the application-layer controller used for
 * whisker-based tactile flight experiments on Crazyflie.
 *
 * The controller supports multiple onboard modes, including:
 * - Finite-state control
 * - Tactile drift online recurrent compensation
 * - MLP  and Kalman Filter for depth estimation
 * - GPIS for active exporation
 * - Other tools
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "commander.h"

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"

#include "log.h"
#include "param.h"
#include <math.h>
#include "usec_time.h"

#include "CF_whiskers_onboard.h"

#define DEBUG_MODULE "CF"

static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;


  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;


  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;

  setpoint->velocity_body = true;
}


static int8_t stateOuterLoop = 0;
static int8_t statemlp = 0;
static int8_t statekf = 0;
static int8_t stateexp = 0;
static int8_t stategpis = 0;
StateCF stateInnerLoop = hover;
StateWhisker statewhisker;

// Some wallfollowing parameters and logging
static float MIN_THRESHOLD1 = 20.0f;
static float MAX_THRESHOLD1 = 100.0f;
static float MIN_THRESHOLD2 = 20.0f;
static float MAX_THRESHOLD2 = 100.0f;
static float scale_1 = 1.0f;
static float scale_2 = 1.0f;
static float offset_1 = 0.0f;
static float offset_2 = 0.0f;
static float maxSpeed = 0.2f;
static float maxTurnRate = 25.0f;

float cmdVelX = 0.0f;
float cmdVelY = 0.0f;
static float cmdHeight = 0.0f;
float cmdAngWDeg = 0.0f;


void appMain()
{
  vTaskDelay(M2T(3000));
  // Getting Logging IDs of the whiskers
  logVarId_t idwhisker1_1 = logGetVarId("Whisker", "Barometer1_1");
  logVarId_t idwhisker1_2 = logGetVarId("Whisker", "Barometer1_2");
  logVarId_t idwhisker1_3 = logGetVarId("Whisker", "Barometer1_3");
  logVarId_t idwhisker2_1 = logGetVarId("Whisker1", "Barometer2_1");
  logVarId_t idwhisker2_2 = logGetVarId("Whisker1", "Barometer2_2");
  logVarId_t idwhisker2_3 = logGetVarId("Whisker1", "Barometer2_3");
  logVarId_t idHeightEstimate = logGetVarId("stateEstimate", "z");

  // Getting the Logging IDs of the state estimates
  logVarId_t idStateYaw = logGetVarId("stateEstimate", "yaw");
  logVarId_t idStateX = logGetVarId("stateEstimate", "x");
  logVarId_t idStateY = logGetVarId("stateEstimate", "y");
  

  // Initialize the wall follower state machine
  bool isFSMInitialized = false; 
  ProcessWhiskerInit(&statewhisker);
  // Intialize the setpoint structure
  setpoint_t setpoint;

  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) 
  {
    const TickType_t xFrequency = pdMS_TO_TICKS(20); // Main control loop at 50 Hz
    const TickType_t xFrequency_GPIS = pdMS_TO_TICKS(260); // Slower scheduling for GPIS updates
    TickType_t xLastWakeTime = xTaskGetTickCount(); // Reference tick for periodic execution
    //DEBUG_PRINT(".");
    float heightEstimate = logGetFloat(idHeightEstimate);

    if (stateOuterLoop == 1) {

      if (!isFSMInitialized) {
            FSMInit(MIN_THRESHOLD1, MAX_THRESHOLD1, MIN_THRESHOLD2, MAX_THRESHOLD2, maxSpeed, maxTurnRate, stateInnerLoop);
            PostCalInit(scale_1, scale_2, offset_1, offset_2);
            isFSMInitialized = true; 
        }

      cmdHeight = 0.3f;
      cmdVelX = 0.0f;
      cmdVelY = 0.0f;
      cmdAngWDeg = 0.0f;

      float whisker1_1 = logGetFloat(idwhisker1_1);
      float whisker1_2 = logGetFloat(idwhisker1_2);
      float whisker1_3 = logGetFloat(idwhisker1_3);
      float whisker2_1 = logGetFloat(idwhisker2_1);
      float whisker2_2 = logGetFloat(idwhisker2_2);
      float whisker2_3 = logGetFloat(idwhisker2_3);
      statewhisker.p_x = logGetFloat(idStateX);
      statewhisker.p_y = logGetFloat(idStateY);
      statewhisker.yaw = logGetFloat(idStateYaw);
      // The wall-following state machine which outputs velocity commands
      float timeNow = usecTimestamp() / 1e6;
      if (statemlp == 0)
      {
        stateInnerLoop = FSM(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker1_2, whisker1_3,
                          whisker2_1, whisker2_2, whisker2_3, &statewhisker, timeNow);
      }
      else if (statemlp == 1)
      {
        if (statekf == 1)
        {
          if (stateexp == 1)
          {
            if (stategpis == 1)
            {
              if (stateInnerLoop == GPIS)
              {
                stateInnerLoop = KFMLPFSM_EXP_GPIS(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker1_2, whisker1_3,
                            whisker2_1, whisker2_2, whisker2_3, &statewhisker, timeNow);
                vTaskDelayUntil(&xLastWakeTime, xFrequency_GPIS);
                xLastWakeTime = xTaskGetTickCount();
              }
              else
              {
                stateInnerLoop = KFMLPFSM_EXP_GPIS(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker1_2, whisker1_3,
                            whisker2_1, whisker2_2, whisker2_3, &statewhisker, timeNow);
              }
            }
            else
            {
              stateInnerLoop = KFMLPFSM_EXP(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker1_2, whisker1_3,
                            whisker2_1, whisker2_2, whisker2_3, &statewhisker, timeNow);
            }
            
          }
          else
          {
            stateInnerLoop = KFMLPFSM(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker1_2, whisker1_3,
                            whisker2_1, whisker2_2, whisker2_3, &statewhisker, timeNow);
          }

        }
        else
        {
          stateInnerLoop = MLPFSM(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker1_2, whisker1_3,
                            whisker2_1, whisker2_2, whisker2_3, &statewhisker, timeNow);
        }
      }
      else if (statemlp == 2)
      {
        stateInnerLoop = FSM_data(&cmdVelX, &cmdVelY, &cmdAngWDeg, whisker1_1, whisker1_2, whisker1_3,
                          whisker2_1, whisker2_2, whisker2_3, &statewhisker, timeNow);
      }

      if (1) 
      {
        setHoverSetpoint(&setpoint, cmdVelX, cmdVelY, cmdHeight, cmdAngWDeg);
        commanderSetSetpoint(&setpoint, 3);
      }
    }
    else
    {
      if (stateOuterLoop == 2)
      { 
        setHoverSetpoint(&setpoint, 0.0f, 0.0f, 0.1f, 0);
        commanderSetSetpoint(&setpoint, 3);
        if (heightEstimate < 0.11f)
        {
          memset(&setpoint, 0, sizeof(setpoint_t));
          commanderSetSetpoint(&setpoint, 3);
        }
      }
    }
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

PARAM_GROUP_START(app)
PARAM_ADD(PARAM_UINT8, stateOuterLoop, &stateOuterLoop)
PARAM_ADD(PARAM_UINT8, statemlp, &statemlp)
PARAM_ADD(PARAM_UINT8, statekf, &statekf)
PARAM_ADD(PARAM_UINT8, stateexp, &stateexp)
PARAM_ADD(PARAM_UINT8, stategpis, &stategpis)
PARAM_ADD(PARAM_FLOAT, MIN_THRESHOLD1, &MIN_THRESHOLD1)
PARAM_ADD(PARAM_FLOAT, MAX_THRESHOLD1, &MAX_THRESHOLD1)
PARAM_ADD(PARAM_FLOAT, MIN_THRESHOLD2, &MIN_THRESHOLD2)
PARAM_ADD(PARAM_FLOAT, MAX_THRESHOLD2, &MAX_THRESHOLD2)
PARAM_ADD(PARAM_FLOAT, scale_1, &scale_1)
PARAM_ADD(PARAM_FLOAT, scale_2, &scale_2)
PARAM_ADD(PARAM_FLOAT, offset_1, &offset_1)
PARAM_ADD(PARAM_FLOAT, offset_2, &offset_2)
PARAM_ADD(PARAM_FLOAT, maxSpeed, &maxSpeed)
PARAM_ADD(PARAM_FLOAT, maxTurnRate, &maxTurnRate)
PARAM_GROUP_STOP(app)

LOG_GROUP_START(app)
LOG_ADD(LOG_FLOAT, PreprocessWhisker1_1, &statewhisker.whisker1_1)
LOG_ADD(LOG_FLOAT, PreprocessWhisker1_2, &statewhisker.whisker1_2)
LOG_ADD(LOG_FLOAT, PreprocessWhisker1_3, &statewhisker.whisker1_3)
LOG_ADD(LOG_FLOAT, PreprocessWhisker2_1, &statewhisker.whisker2_1)
LOG_ADD(LOG_FLOAT, PreprocessWhisker2_2, &statewhisker.whisker2_2)
LOG_ADD(LOG_FLOAT, PreprocessWhisker2_3, &statewhisker.whisker2_3)
LOG_ADD(LOG_FLOAT, MlpOutput1, &statewhisker.mlpoutput_1)
LOG_ADD(LOG_FLOAT, MlpOutput2, &statewhisker.mlpoutput_2)
LOG_ADD(LOG_FLOAT, KFMlpOutput1, &statewhisker.KFoutput_1)
LOG_ADD(LOG_FLOAT, KFMlpOutput2, &statewhisker.KFoutput_2)
LOG_ADD(LOG_UINT8, stateInnerLoop, &stateInnerLoop)
LOG_ADD(LOG_UINT8, stateOuterLoop, &stateOuterLoop)
LOG_GROUP_STOP(app)