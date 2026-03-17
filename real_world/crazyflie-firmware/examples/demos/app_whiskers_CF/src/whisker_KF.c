/*
 * whisker_KF.c
 *
 *  Created on: Sep 9, 2024
 *      Author: Chaoxiang Ye

 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CF_whiskers_onboard.h"
#include "whisker_KF.h"
#include "physicalConstants.h"
#include "debug.h"

// initialize kf
void KF_init(KalmanFilterWhisker* kf, float initial_state1, float initial_state2, float initial_position[], 
             float initial_yaw, float initial_covariance, 
             float process_noise, float measurement_noise) 
{
    kf->x_pre1 = 240 - initial_state1;
    kf->x_pre2 = 240 - initial_state2;
    kf->p_x_last = initial_position[0] * 1000;
    kf->p_y_last = -(initial_position[1] * 1000);
    kf->theta_last = initial_yaw;

    kf->P1 = initial_covariance;
    kf->P2 = initial_covariance;
    kf->Q1 = process_noise;
    kf->Q2 = process_noise;
    kf->R = measurement_noise;
}


// prediction step
void KF_predict(KalmanFilterWhisker* kf, float position[], float yaw) 
{
    float relative_rad = 0;
    float wall_rad = (kf->theta_last * (M_PI_F / 180.0f)) - relative_rad;
    float p_x = position[0] * 1000;
    float p_y = - (position[1] * 1000);
    float relative_distance = sqrtf(powf(p_y - kf->p_y_last, 2) + powf(p_x - kf->p_x_last, 2));
    float new_relative_rad = (yaw * (M_PI_F / 180.0f)) - wall_rad;

    // predict state
    float distance_towall1 = (relative_distance * sinf(wall_rad - atan2f((p_x - kf->p_x_last), (p_y - kf->p_y_last)))) / cosf(relative_rad) + kf->x_pre1;
    float distance_towall2 = (relative_distance * sinf(wall_rad - atan2f((p_x - kf->p_x_last), (p_y - kf->p_y_last)))) / cosf(relative_rad) + kf->x_pre2;
    kf->x_pre1 = (distance_towall1 * cosf(relative_rad)) / cosf(new_relative_rad);
    kf->x_pre2 = (distance_towall2 * cosf(relative_rad)) / cosf(new_relative_rad);
    
    // predict covariance
    kf->x_pre1 = (distance_towall1 * cosf(relative_rad)) / cosf(new_relative_rad);
    kf->x_pre2 = (distance_towall2 * cosf(relative_rad)) / cosf(new_relative_rad);
    
    kf->P1 = kf->P1 + kf->Q1;
    kf->P2 = kf->P2 + kf->Q2;
    // update
    kf->theta_last = yaw;
    kf->p_x_last = p_x;
    kf->p_y_last = p_y;
}

void KF_update(KalmanFilterWhisker* kf,  float z1, float z2) 
{
    
    float P1_plus_R = kf->P1 + kf->R;
    float P2_plus_R = kf->P2 + kf->R;
    float K1 = kf->P1 / P1_plus_R;
    float K2 = kf->P2 / P2_plus_R;

    // update
    kf->x_pre1 += K1 * (z1 - kf->x_pre1);
    kf->x_pre2 += K2 * (z2 - kf->x_pre2);

    // update covariance
    kf->P1 *= (1 - K1);
    kf->P2 *= (1 - K2);
}


// get current state
void KF_get_estimate(KalmanFilterWhisker* kf, float* estimate1, float* estimate2) 
{
    *estimate1 = 240 - kf->x_pre1;
    *estimate2 = 240 - kf->x_pre2;
}


void KF_data_receive(StateWhisker *statewhisker, KalmanFilterWhisker *kf) 
{
    // predict
    KF_predict(kf, (float[]){statewhisker->p_x, statewhisker->p_y}, statewhisker->yaw);
    
    // update
    KF_update(kf, 240 - statewhisker->mlpoutput_1,240 - statewhisker->mlpoutput_2);
    
    // save
    KF_get_estimate(kf, &statewhisker->KFoutput_1, &statewhisker->KFoutput_2);
}