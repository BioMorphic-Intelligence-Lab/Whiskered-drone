/*
 * GPIS.c
 *
 *  Created on: Sep 9, 2024
 *      Author: Chaoxiang Ye

 */

#include "GPIS.h"
#include <stdio.h>
#include <math.h>
#include "arm_math.h"
#include "debug.h"



float inverse_multiquadric_kernel(const float *X, const float *Y, int dim, float c) 
{
    float dist_sq = 0.0f;
    for (int i = 0; i < dim; ++i) 
    {
        dist_sq += (X[i] - Y[i]) * (X[i] - Y[i]);
    }
    return 1.0f / sqrtf(dist_sq + c * c);
}

void gp_fit(GaussianProcess *gp, const float *X_train, const float *y_train, int train_size) 
{
    
    if (train_size > MAX_TRAIN_SIZE) {
        DEBUG_PRINT("Training size exceeds maximum limit.\n");
        return;
    }

    gp->train_size = train_size;
    gp->kernel = inverse_multiquadric_kernel;
    gp->alpha = 1e-2; 

    for (int i = 0; i < train_size; ++i) 
    {
        gp->X_train[i * 2] = X_train[i * 2];
        gp->X_train[i * 2 + 1] = X_train[i * 2 + 1];
        gp->y_train[i] = y_train[i];
    }

    
    float K_matrix_data[MAX_TRAIN_SIZE * MAX_TRAIN_SIZE]; 
    for (int i = 0; i < train_size; ++i) 
    {
        for (int j = 0; j < train_size; ++j) 
        {
            K_matrix_data[i * train_size + j] = gp->kernel(&gp->X_train[i * 2], &gp->X_train[j * 2], 2, 1.0f);
        }
        K_matrix_data[i * train_size + i] += gp->alpha; 
    }

    
    arm_matrix_instance_f32 K_matrix;
    arm_matrix_instance_f32 K_inv;
    arm_mat_init_f32(&K_matrix, train_size, train_size, K_matrix_data);
    arm_mat_init_f32(&K_inv, train_size, train_size, gp->K_inv);
    
    
    if (arm_mat_inverse_f32(&K_matrix, &K_inv) != ARM_MATH_SUCCESS) 
    {
        DEBUG_PRINT("Matrix is singular and cannot be inverted.\n");
    }
}


void gp_predict(const GaussianProcess *gp, const float *X_test, int test_size, float *y_pred, float *y_std) 
{
    
    arm_matrix_instance_f32 K_trans, K_test, K_inv_instance, y_train, y_mean;

    
    float K_trans_data[MAX_TRAIN_SIZE * test_size]; 
    arm_mat_init_f32(&K_trans, test_size, gp->train_size, K_trans_data);

    
    for (int i = 0; i < test_size; ++i) 
    {
        for (int j = 0; j < gp->train_size; ++j) 
        {
            K_trans.pData[i * gp->train_size + j] = gp->kernel(&X_test[i * 2], &gp->X_train[j * 2], 2, 1.0f);
        }
    }

    
    arm_mat_init_f32(&K_inv_instance, gp->train_size, gp->train_size, (float*)gp->K_inv);
    arm_mat_init_f32(&y_train, gp->train_size, 1, (float*)gp->y_train);

    arm_mat_init_f32(&y_mean, test_size, 1, y_pred); 

    
    arm_matrix_instance_f32 K_trans_K_inv;
    float K_trans_K_inv_data[MAX_TRAIN_SIZE * test_size]; 
    arm_mat_init_f32(&K_trans_K_inv, test_size, gp->train_size, K_trans_K_inv_data);

    arm_mat_mult_f32(&K_trans, &K_inv_instance, &K_trans_K_inv);  // K_trans * K_inv
    arm_mat_mult_f32(&K_trans_K_inv, &y_train, &y_mean);  // K_trans_K_inv * y_train

    
    float K_test_data[test_size*test_size];
    arm_mat_init_f32(&K_test, test_size, test_size, K_test_data);
    
   
    for (int i = 0; i < test_size; ++i) 
    {
        for (int j = 0; j < test_size; ++j) 
        {
            K_test.pData[i * test_size + j] = gp->kernel(&X_test[i * 2], &X_test[j * 2], 2, 1.0f);
        }
    }

    
    arm_matrix_instance_f32 K_trans_T;
    float K_trans_T_data[MAX_TRAIN_SIZE*test_size]; 
    arm_mat_init_f32(&K_trans_T, gp->train_size, test_size, K_trans_T_data);
    arm_mat_trans_f32(&K_trans, &K_trans_T); 

    
    arm_matrix_instance_f32 K_trans_K_inv_K_transT;
    float K_trans_K_inv_K_transT_data[1]; 
    arm_mat_init_f32(&K_trans_K_inv_K_transT, test_size, test_size, K_trans_K_inv_K_transT_data);
    
    arm_mat_mult_f32(&K_trans_K_inv, &K_trans_T, &K_trans_K_inv_K_transT); // K_trans_K_inv * K_trans

    // y_var = K_test - K_trans_K_inv_K_transT
    arm_matrix_instance_f32 y_var;
    float y_var_data[test_size * test_size]; 
    arm_mat_init_f32(&y_var, test_size, test_size, y_var_data);
    
    arm_mat_sub_f32(&K_test, &K_trans_K_inv_K_transT, &y_var); // K_test - K_trans_K_inv_K_transT

    
    for (int i = 0; i < test_size; ++i) {
        y_std[i] = sqrtf(y_var.pData[i * test_size + i]);
    }
    
    
}
