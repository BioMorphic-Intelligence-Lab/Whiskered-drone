// GPIS.h
#ifndef GPIS_H
#define GPIS_H

#include <stdlib.h>
#define MAX_TRAIN_SIZE 50  // 最大训练样本数量
#define MAX_TEST_SIZE 100    // 最大测试样本数量

typedef struct {
    int train_size;
    float X_train[MAX_TRAIN_SIZE * 2]; // 使用静态数组
    float y_train[MAX_TRAIN_SIZE]; // 使用静态数组
    float K_inv[MAX_TRAIN_SIZE * MAX_TRAIN_SIZE]; // 使用静态数组
    float (*kernel)(const float*, const float*, int, float);
    float alpha;
} GaussianProcess;

void gp_fit(GaussianProcess *gp, const float *X_train, const float *y_train, int train_size);
void gp_predict(const GaussianProcess *gp, const float *X_test, int test_size, float *y_pred, float *y_std);
void gp_free(GaussianProcess *gp);

#endif // GPIS_H