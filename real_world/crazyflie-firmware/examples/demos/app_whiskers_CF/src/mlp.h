#ifndef SRC_MLP_H_
#define SRC_MLP_H_
#include <stdint.h>
#include <stdbool.h>


#define INPUT_SIZE 3
#define HIDDEN_SIZE_1 32 
#define HIDDEN_SIZE_2 32 
#define HIDDEN_SIZE_3 32 

#include <arm_math.h>


void dis_net(StateWhisker *statewhisker, float scale_1, float scale_2, float offset_1, float offset_2);

#endif /* SRC_MLP_H_ */
