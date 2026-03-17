#define DEBUG_MODULE "WHISKER"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "debug.h"
#include "log.h"
#include "deck.h"
#include "uart1.h"
#include "FreeRTOS.h"
#include "task.h"
#include "system.h"

#define MAX_MESSAGE_SIZE 48

static uint8_t isInit = 0;
// uint32_t loggedTimestamp = 0;
float barometer1_1, barometer1_2, barometer1_3, temperature1_1, temperature1_2, temperature1_3;
// const float coeffs1[] = {11727860.1289f, -1411009.9414f, 56617.3234f, -757.6631f};
// const float coeffs2[] = {-91572245.7666f, 10553623.8619f, -405241.7287f, 5184.3470f};
// const float coeffs3[] = {-82609270.8952f, 9801474.2071f, -387569.7598f, 5107.4214f};

// float compensate(float pressure, float temp, const float *coeffs) {
//     float t2 = temp * temp;
//     float t3 = t2 * temp;
//     float drift = coeffs[0] + coeffs[1]*temp + coeffs[2]*t2 + coeffs[3]*t3;
//     return pressure - drift;
// }

void readSerial1() {
    char buf[32];
    char c;
    int readCount = 0;

    float p1 = 0, p2 = 0, p3 = 0;
    float t1 = 0, t2 = 0, t3 = 0;

    while (readCount < 6) {
        int i = 0;

        // Read a line
        while (i < sizeof(buf) - 1) {
            if (!uart1GetDataWithDefaultTimeout(&c)) return;
            if (c == '\n') {
                buf[i] = '\0';
                break;
            }
            buf[i++] = c;
        }

        // Check label and parse value
        if (strncmp(buf, "P1:", 3) == 0) {
            p1 = atof(buf + 3);
            readCount++;
        } else if (strncmp(buf, "P2:", 3) == 0) {
            p2 = atof(buf + 3);
            readCount++;
        } else if (strncmp(buf, "P3:", 3) == 0) {
            p3 = atof(buf + 3);
            readCount++;
        } else if (strncmp(buf, "T1:", 3) == 0) {
            t1 = atof(buf + 3);
            readCount++;
        } else if (strncmp(buf, "T2:", 3) == 0) {
            t2 = atof(buf + 3);
            readCount++;
        } else if (strncmp(buf, "T3:", 3) == 0) {
            t3 = atof(buf + 3);
            readCount++;
        }
    }

    // Apply compensation
    barometer1_1 = p1;
    barometer1_2 = p2;
    barometer1_3 = p3;
    temperature1_1 = t1;
    temperature1_2 = t2;
    temperature1_3 = t3;
}




void WhiskerTask(void *param) {
    systemWaitStart();
    while (1) {
        readSerial1();
    }
}

static void WhiskerInit() {
    DEBUG_PRINT("Initialize driver\n");

    uart1Init(115200);

    xTaskCreate(WhiskerTask, WHISKER_TASK_NAME, WHISKER_TASK_STACKSIZE, NULL,
                WHISKER_TASK_PRI, NULL);

    isInit = 1;
}

static bool WhiskerTest() {
    return isInit;
}

static const DeckDriver WhiskerDriver = {
        .name = "Whisker",
        .init = WhiskerInit,
        .test = WhiskerTest,
        .usedPeriph = DECK_USING_UART1,
};

DECK_DRIVER(WhiskerDriver);


/**
 * Logging variables for the Whisker
 */
LOG_GROUP_START(Whisker)
LOG_ADD(LOG_FLOAT, Barometer1_1, &barometer1_1)
LOG_ADD(LOG_FLOAT, Barometer1_2, &barometer1_2)
LOG_ADD(LOG_FLOAT, Barometer1_3, &barometer1_3)
LOG_ADD(LOG_FLOAT, Temperature1_1, &temperature1_1)
LOG_ADD(LOG_FLOAT, Temperature1_2, &temperature1_2)
LOG_ADD(LOG_FLOAT, Temperature1_3, &temperature1_3)
LOG_GROUP_STOP(Whisker) 