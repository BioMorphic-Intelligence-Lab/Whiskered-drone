/*
 * CF_whiskers_onboard.c
 *
 *  Created on: Sep 9, 2024
 *      Author: Chaoxiang Ye

 */

#include "CF_whiskers_onboard.h"
#include <math.h>
#include <stdlib.h>
#include "mlp.h"
#include "whisker_KF.h"
#include "debug.h"
#include "log.h"
#include "GPIS.h"
#include "physicalConstants.h"
#include "utlz.h"
#include "FreeRTOS.h"
#include "time.h"
#define DATA_SIZE 100

static float firstRun = false;
static float firstRunPreprocess = false;
static float maxSpeed = 0.2f;
static float maxTurnRate = 25.0f;
static float direction = 1.0f;
static float CF_THRESHOLD1 = 20.0f;
static float CF_THRESHOLD2 = 20.0f;
static float MIN_THRESHOLD1 = 20.0f;
static float MAX_THRESHOLD1 = 100.0f;
static float MIN_THRESHOLD2 = 20.0f;
static float MAX_THRESHOLD2 = 100.0f;
static float MAX_FILTERTHRESHOLD1 = 20.0f;
static float MAX_FILTERTHRESHOLD2 = 20.0f;
static float StartTime;
static const float waitForStartSeconds = 5.0f;
static float stateStartTime;
static int is_KF_initialized = 0;
static int CF_count = 350;
static int backward_count = 100;
static int rotate_count = 180;
static int rotation_time = 0;
int8_t GPIS_label = 0;
// GP params
GaussianProcess gp_model; 
static int current_train_index = 0;  
static float X_train[MAX_TRAIN_SIZE * 2];  
static float y_train[MAX_TRAIN_SIZE];   
static KalmanFilterWhisker kf;
static float process_noise = 0.01;
static float measurement_noise = 1;
static float initial_covariance = 50;
static float scale_1 = 1.0f;
static float scale_2 = 1.0f;
static float offset_1 = 0.0f;
static float offset_2 = 0.0f;

static StateCF stateCF = hover;
float timeNow = 0.0f;

void ProcessWhiskerInit(StateWhisker *statewhisker) 
{
    for (int i = 0; i < 6; i++) 
    {
        statewhisker->zi[i][0] = 0.0f;  // Initialize filter state
        statewhisker->zi[i][1] = 0.0f;
    }
    
    statewhisker->count = 0;
    statewhisker->preprocesscount = 0;
    
    // 1st order butterworth filter 0.05-1hz
    statewhisker->b[0] = 0.05639124;
    statewhisker->b[1] = 0.0f;
    statewhisker->b[2] = -0.05639124;
    statewhisker->a[0] = 1.0f;
    statewhisker->a[1] = -1.88647164;
    statewhisker->a[2] = 0.88721752;
    firstRunPreprocess = true;
    DEBUG_PRINT("Initialize preprocessing parameters.\n");
}

void update_statistics(StateWhisker *statewhisker, float data, int index) 
{
    float x = (float)statewhisker->count;
    statewhisker->sum_x[index] += x;
    statewhisker->sum_y[index] += data;
    statewhisker->sum_x_squared[index] += x * x;
    statewhisker->sum_xy[index] += x * data;
}


void calculate_parameters(StateWhisker *statewhisker) 
{
    int n = DATA_SIZE;
    for (int i = 0; i < 6; i++) 
    {
        statewhisker->slopes[i] = (n * statewhisker->sum_xy[i] - statewhisker->sum_x[i] * statewhisker->sum_y[i]) /
                             (n * statewhisker->sum_x_squared[i] - statewhisker->sum_x[i] * statewhisker->sum_x[i]);
        statewhisker->intercepts[i] = (statewhisker->sum_y[i] - statewhisker->slopes[i] * statewhisker->sum_x[i]) / n;
    }
}


void apply_bandpass_filter(float data, float *zi, float *filtered_data, float *b, float *a) 
{
    float input = data;
    float output = b[0] * input + zi[0];
    zi[0] = b[1] * input - a[1] * output + zi[1];
    zi[1] = b[2] * input - a[2] * output;
    *filtered_data = output;
}



void process_data(StateWhisker *statewhisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) 
{
    float residuals[6];
    residuals[0] = whisker1_1 - (statewhisker->slopes[0] * statewhisker->count + statewhisker->intercepts[0]);
    residuals[1] = whisker1_2 - (statewhisker->slopes[1] * statewhisker->count + statewhisker->intercepts[1]);
    residuals[2] = whisker1_3 - (statewhisker->slopes[2] * statewhisker->count + statewhisker->intercepts[2]);
    residuals[3] = whisker2_1 - (statewhisker->slopes[3] * statewhisker->count + statewhisker->intercepts[3]);
    residuals[4] = whisker2_2 - (statewhisker->slopes[4] * statewhisker->count + statewhisker->intercepts[4]);
    residuals[5] = whisker2_3 - (statewhisker->slopes[5] * statewhisker->count + statewhisker->intercepts[5]);

    apply_bandpass_filter(residuals[0], statewhisker->zi[0], &statewhisker->whisker1_1, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[1], statewhisker->zi[1], &statewhisker->whisker1_2, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[2], statewhisker->zi[2], &statewhisker->whisker1_3, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[3], statewhisker->zi[3], &statewhisker->whisker2_1, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[4], statewhisker->zi[4], &statewhisker->whisker2_2, statewhisker->b, statewhisker->a);
    apply_bandpass_filter(residuals[5], statewhisker->zi[5], &statewhisker->whisker2_3, statewhisker->b, statewhisker->a);
}

void ProcessDataReceived(StateWhisker *statewhisker, float whisker1_1, float whisker1_2, float whisker1_3, float whisker2_1, float whisker2_2, float whisker2_3) 
{
    if (firstRunPreprocess)
    {
        update_statistics(statewhisker, whisker1_1, 0);
        update_statistics(statewhisker, whisker1_2, 1);
        update_statistics(statewhisker, whisker1_3, 2);
        update_statistics(statewhisker, whisker2_1, 3);
        update_statistics(statewhisker, whisker2_2, 4);
        update_statistics(statewhisker, whisker2_3, 5);
        statewhisker->count++;
        if (statewhisker->count == DATA_SIZE) 
        {
            calculate_parameters(statewhisker);
            for (int i = 0; i < 6; i++) 
            {
                statewhisker->sum_x[i] = 0.0f;
                statewhisker->sum_y[i] = 0.0f;
                statewhisker->sum_x_squared[i] = 0.0f;
                statewhisker->sum_xy[i] = 0.0f;
            }
            firstRunPreprocess = false;
            DEBUG_PRINT("First Apply linear fitting parameters.\n");
        }
    }
    else 
    {   
        update_statistics(statewhisker, whisker1_1, 0);
        update_statistics(statewhisker, whisker1_2, 1);
        update_statistics(statewhisker, whisker1_3, 2);
        update_statistics(statewhisker, whisker2_1, 3);
        update_statistics(statewhisker, whisker2_2, 4);
        update_statistics(statewhisker, whisker2_3, 5);
        statewhisker->count++;
        statewhisker->preprocesscount++;
        process_data(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > MAX_FILTERTHRESHOLD1 || statewhisker->whisker2_1 > MAX_FILTERTHRESHOLD2)
        {
            for (int i = 0; i < 6; i++) 
            {
                statewhisker->sum_x[i] = 0.0f;
                statewhisker->sum_y[i] = 0.0f;
                statewhisker->sum_x_squared[i] = 0.0f;
                statewhisker->sum_xy[i] = 0.0f;
            }
            statewhisker->preprocesscount = 0;
        }
        else if (statewhisker->preprocesscount == DATA_SIZE) 
        {
            calculate_parameters(statewhisker);
            for (int i = 0; i < 6; i++) 
            {
                statewhisker->sum_x[i] = 0.0f;
                statewhisker->sum_y[i] = 0.0f;
                statewhisker->sum_x_squared[i] = 0.0f;
                statewhisker->sum_xy[i] = 0.0f;
                statewhisker->zi[i][0] = 0.0f;  // Initialize filter state
                statewhisker->zi[i][1] = 0.0f;
            }
            statewhisker->preprocesscount = 0;
            DEBUG_PRINT("Apply linear fitting parameters.\n");
        }
    }
}

void FSMInit(float MIN_THRESHOLD1_input, float MAX_THRESHOLD1_input, 
             float MIN_THRESHOLD2_input, float MAX_THRESHOLD2_input, 
             float maxSpeed_input, float maxTurnRate_input, StateCF initState)
{
  MIN_THRESHOLD1 = MIN_THRESHOLD1_input;
  MAX_THRESHOLD1 = MAX_THRESHOLD1_input;
  MIN_THRESHOLD2 = MIN_THRESHOLD2_input;
  MAX_THRESHOLD2 = MAX_THRESHOLD2_input;
  maxSpeed = maxSpeed_input;
  maxTurnRate = maxTurnRate_input;
  firstRun = true;
  stateCF = initState;
  CF_count = 350;
  DEBUG_PRINT("Initialize FSM parameters.\n");
}

void PostCalInit(float scale_1_input, float scale_2_input, 
             float offset_1_input, float offset_2_input)
{
  scale_1 = scale_1_input;
  scale_2 = scale_2_input;
  offset_1 = offset_1_input;
  offset_2= offset_2_input;

  DEBUG_PRINT("Initialize Post calibrated parameters.\n");
}

static StateCF transition(StateCF newState)
{
  stateStartTime = timeNow;
  return newState;
}

StateCF FSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
        }
        break;
    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}


StateCF MLPFSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        CF_count--;
        if (CF_count < 0)
        {
            stateCF = transition(backward);
            DEBUG_PRINT("CF finished. Flyingbackward.\n");
        }
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {   
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            DEBUG_PRINT("%f\n", (double)scale_1);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
            
        }else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->mlpoutput_2 = 0.0f;
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->mlpoutput_1 = 0.0f; 
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");

            statewhisker->mlpoutput_1 = 0.0f; 
            statewhisker->mlpoutput_2 = 0.0f;
            DEBUG_PRINT("Exiting CF state, resources released.\n");
        }
        break;

    case backward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        break;
    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    case backward:
        cmdVelXTemp = -1.0f * maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}

StateCF FSM_data(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        CF_count--;
        if (CF_count < 0)
        {
            stateCF = transition(backward);
            DEBUG_PRINT("CF finished. Flyingbackward.\n");
        }
        else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
        }
        break;

    case backward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        break;

    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->whisker1_1 < MIN_THRESHOLD1 && statewhisker->whisker2_1 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && statewhisker->whisker2_1 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->whisker1_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->whisker1_1 && statewhisker->whisker1_1 > MIN_THRESHOLD1 && statewhisker->whisker2_1 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->whisker1_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->whisker2_1 && statewhisker->whisker2_1 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    case backward:
        cmdVelXTemp = -1.0f * maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}

StateCF KFMLPFSM(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;

    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);

        if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {        
            if (!is_KF_initialized)
            {
                dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
                KF_init(&kf, statewhisker->mlpoutput_1, statewhisker->mlpoutput_2, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw, initial_covariance, process_noise, measurement_noise);//here need a interface
                is_KF_initialized = 1; // 标记已初始化
                statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
                statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            }
            else
            {
                dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
                KF_data_receive(statewhisker, &kf);
            }

        }else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_2 = 0.0f;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 2 loses contact. Reset KF.\n");
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 1 loses contact. Reset KF.\n");
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
            is_KF_initialized = 0;
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            DEBUG_PRINT("Exiting CF state, resources released.\n");
        }
        break;
    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}

StateCF KFMLPFSM_EXP(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;
    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }
    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;
    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;
    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        CF_count--;
        if (CF_count < 0)
        {
            stateCF = transition(backward);
            DEBUG_PRINT("CF finished. Flyingbackward.\n");
            backward_count = 100;
        }
        else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {        
            if (!is_KF_initialized)
            {
                dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
                KF_init(&kf, statewhisker->mlpoutput_1, statewhisker->mlpoutput_2, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw, initial_covariance, process_noise, measurement_noise);//here need a interface
                is_KF_initialized = 1; 
                statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
                statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            }
            else
            {
                dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
                KF_data_receive(statewhisker, &kf);
            }
        }else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_2 = 0.0f;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 2 loses contact. Reset KF.\n");
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 1 loses contact. Reset KF.\n");
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
            is_KF_initialized = 0;
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            DEBUG_PRINT("Exiting CF state, resources released.\n");
        }
        break;
    case backward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        backward_count--;
        if (backward_count < 0)
        {
            stateCF = transition(rotate);
            DEBUG_PRINT("Backward finished. Starting rotating.\n");
            rotate_count = 180;
        }
        break;
    case rotate:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        rotate_count--;
        if (rotate_count <0)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Rotate finished. Starting flying forward.\n");
            CF_count = 125;
        }
        break;
    default:
        stateCF = transition(forward);
        break;
    }
    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;
    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;
    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;
    case CF:
        if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;
    case backward:
        cmdVelXTemp = -1.0f * maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;
    case rotate:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = -1.0f * maxTurnRate;
        break;
    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }
    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;
    return stateCF;
}

StateCF KFMLPFSM_EXP_GPIS(float *cmdVelX, float *cmdVelY, float *cmdAngW, float whisker1_1, float whisker1_2, float whisker1_3, 
            float whisker2_1, float whisker2_2, float whisker2_3, StateWhisker *statewhisker, float timeOuter)
{
    timeNow = timeOuter;
    GPIS_label = 2;
    if (firstRun)
    {
        firstRun = false;
        StartTime = timeNow;
    }

    // Handle state transitions
    switch (stateCF)
    {
    case hover:
        if (timeNow - StartTime >= waitForStartSeconds)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Hover complete. Starting forward.\n");
        }
        break;

    case forward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        if (statewhisker->whisker1_1 > CF_THRESHOLD1 || statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            stateCF = transition(CF);
            DEBUG_PRINT("Obstacles encountered. Starting contour tracking.\n");
        }
        break;

    case CF:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        CF_count--;
        if (CF_count < 0)
        {
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->mlpoutput_2 = 0.0f;
            is_KF_initialized = 0;
            stateCF = transition(backward);
            DEBUG_PRINT("CF finished. Flyingbackward.\n");
            backward_count = 100;
        }
        else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {        
            if (!is_KF_initialized)
            {
                dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
                KF_init(&kf, statewhisker->mlpoutput_1, statewhisker->mlpoutput_2, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw, initial_covariance, process_noise, measurement_noise);//here need a interface
                is_KF_initialized = 1; 
                statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
                statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
                if (current_train_index < MAX_TRAIN_SIZE && CF_count % 40 == 0) 
                {
              
                    X_train[current_train_index * 2] = (240.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f)/1000.0f * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x;   
                    X_train[current_train_index * 2 + 1] =  (statewhisker->p_y) + (240.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f)/1000.0f  * sinf(statewhisker->yaw * (M_PI_F / 180.0f));
                    DEBUG_PRINT("%f,%f\n", (double)X_train[current_train_index * 2], (double)X_train[current_train_index * 2 + 1]);
                    
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  
                    GPIS_label = 0;
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
            }
            else
            {
                dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
                KF_data_receive(statewhisker, &kf);
                if (current_train_index < MAX_TRAIN_SIZE && CF_count % 40 == 0) 
                {
                    
                    X_train[current_train_index * 2] = (240.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f)/1000.0f  * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x;   
                    X_train[current_train_index * 2 + 1] =  (statewhisker->p_y) + (240.0f - (statewhisker->KFoutput_1 + statewhisker->KFoutput_2)/2.0f)/1000.0f  * sinf(statewhisker->yaw * (M_PI_F / 180.0f)); 
                    DEBUG_PRINT("%f,%f\n", (double)X_train[current_train_index * 2], (double)X_train[current_train_index * 2 + 1]);
                   
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  
                    GPIS_label = 0;
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
            }

        }else if (statewhisker->whisker1_1 > CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->KFoutput_1 = statewhisker->mlpoutput_1;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_2 = 0.0f;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 2 loses contact. Reset KF.\n");
            if (current_train_index < MAX_TRAIN_SIZE && CF_count % 40 == 0) 
                {
                    
                    X_train[current_train_index * 2] = (240.0f - statewhisker->KFoutput_1)/1000.0f  * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x;   
                    X_train[current_train_index * 2 + 1] = (statewhisker->p_y) + (240.0f - statewhisker->KFoutput_1)/1000.0f * sinf(statewhisker->yaw * (M_PI_F / 180.0f)); 
                    DEBUG_PRINT("%f,%f\n", (double)X_train[current_train_index * 2], (double)X_train[current_train_index * 2 + 1]);
                    
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  
                    GPIS_label = 0;
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 > CF_THRESHOLD2)
        {
            dis_net(statewhisker, scale_1, scale_2, offset_1, offset_2);
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = statewhisker->mlpoutput_2;
            is_KF_initialized = 0;
            DEBUG_PRINT("Whisker 1 loses contact. Reset KF.\n");
            if (current_train_index < MAX_TRAIN_SIZE && CF_count % 40 == 0) 
                {
                    
                    X_train[current_train_index * 2] = (240.0f - statewhisker->KFoutput_2)/1000.0f * cosf(statewhisker->yaw * (M_PI_F / 180.0f)) + statewhisker->p_x;   
                    X_train[current_train_index * 2 + 1] =  (statewhisker->p_y) + (240.0f - statewhisker->KFoutput_2)/1000.0f * sinf(statewhisker->yaw * (M_PI_F / 180.0f)); 
                    DEBUG_PRINT("%f,%f\n", (double)X_train[current_train_index * 2], (double)X_train[current_train_index * 2 + 1]);
                    
                    y_train[current_train_index] = 0.0f;
                    current_train_index++;  
                    GPIS_label = 0;
                    DEBUG_PRINT("Surface Training sample saved. Current index: %d\n", current_train_index);
                }
        }else if (statewhisker->whisker1_1 < CF_THRESHOLD1 && statewhisker->whisker2_1 < CF_THRESHOLD2)
        {   
            CF_count++;
            stateCF = transition(forward);
            DEBUG_PRINT("Lose contact. Flyingforward.\n");
            is_KF_initialized = 0;
            statewhisker->KFoutput_1 = 0.0f;
            statewhisker->KFoutput_2 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            statewhisker->mlpoutput_1 = 0.0f;
            DEBUG_PRINT("Exiting CF state, resources released.\n");
        }
        break;

    case backward:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        backward_count--;
        if (backward_count < 0)
        {
            stateCF = transition(GPIS);
            DEBUG_PRINT("Backward finished. Starting rotating.\n");
            rotate_count = 180;
            if (current_train_index < MAX_TRAIN_SIZE) 
                {
                    
                    X_train[current_train_index * 2] = statewhisker->p_x;   
                    X_train[current_train_index * 2 + 1] = (statewhisker->p_y); 
                    DEBUG_PRINT("%f,%f\n", (double)X_train[current_train_index * 2], (double)X_train[current_train_index * 2 + 1]);

                    
                    y_train[current_train_index] = -1.0f;
                    current_train_index++;
                    GPIS_label = -1;
                    DEBUG_PRINT("Inside Training sample saved. Current index: %d\n", current_train_index);
                }
        }
        break;

    case rotate:
        ProcessDataReceived(statewhisker, whisker1_1, whisker1_2, whisker1_3, whisker2_1, whisker2_2, whisker2_3);
        rotate_count--;
        if (rotate_count <0)
        {
            stateCF = transition(forward);
            DEBUG_PRINT("Rotate finished. Starting flying forward.\n");
            CF_count = 125;
            rotation_time++;
        }
        break;
    
    case GPIS:
        if (rotation_time < 3)
        {
            stateCF = transition(rotate);
        }

        else if (current_train_index > 0) 
        {
            float time_start = usecTimestamp() / 1e6;
            
            gp_fit(&gp_model, X_train, y_train, current_train_index);
            DEBUG_PRINT("GP model trained with %d training samples.\n", current_train_index);

            int8_t grid_size = 10; 
            float x_min = -2.0f;
            float x_max = 2.0f;
            float y_min = -2.0f;
            float y_max = 2.0f;
            float x_step = (x_max - x_min) / (grid_size - 1);
            float y_step = (y_max - y_min) / (grid_size - 1);

            float y_preds[grid_size * grid_size];
            float y_stds[grid_size * grid_size];

            
            float X_test[2];
            float y_pred[1], y_std[1];
            for (int i = 0; i < grid_size; ++i) 
            {
                for (int j = 0; j < grid_size; ++j) 
                {
                    X_test[0] = x_min + j * x_step;
                    X_test[1] = y_min + i * y_step;
                    gp_predict(&gp_model, X_test, 1, y_pred, y_std);

                    int idx = i * grid_size + j;
                    y_preds[idx] = y_pred[0];
                    y_stds[idx] = y_std[0];
                if (isnan(y_stds[idx]) || isinf(y_stds[idx])) 
                    {
                        DEBUG_PRINT("Invalid y_std value detected at index %d: %f\n", idx, (double)y_stds[idx]);
                    }
                }
            }
            LineSegment lineSegments[grid_size*grid_size];       
            Point orderedContourPoints[grid_size*grid_size]; 
            int segmentCount = 0;
            int orderedPointCount = 0;
            marchingSquares(grid_size, y_preds, y_stds, x_min, x_step, y_min, y_step, lineSegments, &segmentCount);
            connectContourSegments(lineSegments, orderedContourPoints, &segmentCount, &orderedPointCount);
            
            float significant_points[grid_size * grid_size / 2];
            int num_significant_points;
            float curvatures[grid_size * grid_size];
            compute_curvature_kernel(&gp_model, orderedContourPoints, orderedPointCount, curvatures);
            find_high_curvature_clusters_using_curvature(orderedContourPoints, orderedPointCount, curvatures, -1.0f,  significant_points, &num_significant_points);
           
            apply_penalty(orderedContourPoints, orderedPointCount, y_stds, significant_points, num_significant_points, 0.2f);
            
            
            float max_y_std = -FLT_MAX; 
            int max_y_std_index = -1;
            for (int i = 0; i < orderedPointCount; ++i) 
            {
                
                float current_y_std = orderedContourPoints[i].y_std; 
                if (current_y_std > max_y_std) 
                {
                    max_y_std = current_y_std;
                    max_y_std_index = i; 
                }
            }
            float max_x = 0.0f, max_y = 0.0f;
            
            if (max_y_std_index != -1) 
            {
                max_x = orderedContourPoints[max_y_std_index].x; 
                max_y = orderedContourPoints[max_y_std_index].y; 
                DEBUG_PRINT("Max y_std point at (x = %f, y = %f) with y_std = %f\n", (double)max_x, (double)max_y, (double)max_y_std);
            }

            rotate_count = (int)roundf(calculate_rotation_time(statewhisker->p_x, statewhisker->p_y, max_x, max_y, statewhisker->yaw, maxTurnRate, &direction));
            float time_end = usecTimestamp() / 1e6;
            double time_taken = (double)(time_end - time_start);
            DEBUG_PRINT("run time: %f s\n", time_taken);
        }
        stateCF = transition(rotate); 
        firstRunPreprocess = true;
        for (int i = 0; i < 6; i++) 
            {
                statewhisker->sum_x[i] = 0.0f;
                statewhisker->sum_y[i] = 0.0f;
                statewhisker->sum_x_squared[i] = 0.0f;
                statewhisker->sum_xy[i] = 0.0f;
                statewhisker->zi[i][0] = 0.0f;  // Initialize filter state
                statewhisker->zi[i][1] = 0.0f;
            }
        statewhisker->count = 0;
        statewhisker->preprocesscount = 0;

        break;


    default:
        stateCF = transition(forward);
        break;
    }

    // Handle state actions
    float cmdVelXTemp = 0.0f;
    float cmdVelYTemp = 0.0f;
    float cmdAngWTemp = 0.0f;

    switch (stateCF)
    {
    case hover:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case forward:
        cmdVelXTemp = maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case CF:
        if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = -1.0f * maxSpeed;
            cmdAngWTemp = 0.0f;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && statewhisker->KFoutput_2 < MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 > MAX_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = maxTurnRate;
        }
        else if (MAX_THRESHOLD1 > statewhisker->KFoutput_1 && statewhisker->KFoutput_1 > MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && statewhisker->KFoutput_2 > MAX_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else if (statewhisker->KFoutput_1 < MIN_THRESHOLD1 && MAX_THRESHOLD2 > statewhisker->KFoutput_2 && statewhisker->KFoutput_2 > MIN_THRESHOLD2)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = -1.0f * maxSpeed / 2.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = 0.0f;
        }
        break;

    case backward:
        cmdVelXTemp = -1.0f * maxSpeed;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    case rotate:
        if (rotation_time < 3)
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = -1.0f * maxTurnRate;
        }
        else
        {
            cmdVelXTemp = 0.0f;
            cmdVelYTemp = 0.0f;
            cmdAngWTemp = direction * maxTurnRate;
        }
        break;

    case GPIS:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
        break;

    default:
        cmdVelXTemp = 0.0f;
        cmdVelYTemp = 0.0f;
        cmdAngWTemp = 0.0f;
    }

    *cmdVelX = cmdVelXTemp;
    *cmdVelY = cmdVelYTemp;
    *cmdAngW = cmdAngWTemp;

    return stateCF;
}


LOG_GROUP_START(app)
LOG_ADD(LOG_INT8, GpisLabel, &GPIS_label)
LOG_GROUP_STOP(app)