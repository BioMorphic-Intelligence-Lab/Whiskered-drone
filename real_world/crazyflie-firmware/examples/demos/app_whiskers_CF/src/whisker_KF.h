#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

typedef struct {
    float x_pre1;            
    float x_pre2; 
    float p_x_last;        
    float p_y_last;       
    float theta_last;          
    float P1;       
    float P2; 
    float Q1;       
    float Q2; 
    float R;       
} KalmanFilterWhisker;

void KF_init(KalmanFilterWhisker* kf, float initial_state1, float initial_state2, float initial_position[], 
             float initial_yaw, float initial_covariance, 
             float process_noise, float measurement_noise) ;
void KF_predict(KalmanFilterWhisker* kf, float position[], float yaw);
void KF_update(KalmanFilterWhisker* kf,  float z1, float z2) ;
void KF_get_estimate(KalmanFilterWhisker* kf, float* estimate1, float* estimate2);
void KF_data_receive(StateWhisker *statewhisker, KalmanFilterWhisker *kf);

#endif // KALMAN_FILTER_H
