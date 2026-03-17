#define DEBUG_MODULE "WHISKER1"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "debug.h"
#include "log.h"
#include "deck.h"
#include "uart2.h"
#include "FreeRTOS.h"
#include "task.h"
#include "system.h"

#define MAX_MESSAGE_SIZE 48

static uint8_t isInit = 0;
// uint32_t loggedTimestamp = 0;
float barometer2_1, barometer2_2, barometer2_3; 


void readSerial2() {
    char buf[MAX_MESSAGE_SIZE];
    char c = '0';
    char* token = NULL;
    int tokenCount = 0;

    for (int i = 0; i < MAX_MESSAGE_SIZE - 1; i++) {
        if (!uart2GetCharWithDefaultTimeout(&c)) break;

        if (c == '\n') {
            buf[i] = '\0';

            // 使用strtok分割字符串
            token = strtok(buf, ",");
            while (token != NULL && tokenCount < 3) {
                switch (tokenCount) {
                    case 0:
                        barometer2_1 = atof(token);
                        break;
                    case 1:
                        barometer2_2 = atof(token);
                        break;
                    case 2:
                        barometer2_3 = atof(token);
                        break;
                }

                token = strtok(NULL, ",");
                tokenCount++;
            }

            if (tokenCount < 3) {
                DEBUG_PRINT("Error parsing sensor2 data\n");
            }

            break;
        }

        buf[i] = c;

        if (c == ',') {
            // 重置 token 和 tokenCount
            token = NULL;
            tokenCount = 0;
        }
    }
}


void WhiskerTask1(void *param) {
    systemWaitStart();
    while (1) {
        readSerial2();
    }
}

static void WhiskerInit1() {
    DEBUG_PRINT("Initialize driver\n");

    uart2Init(115200);

    xTaskCreate(WhiskerTask1, WHISKER_TASK_NAME1, WHISKER_TASK_STACKSIZE, NULL,
                WHISKER_TASK_PRI, NULL);

    isInit = 1;
}

static bool WhiskerTest1() {
    return isInit;
}

static const DeckDriver WhiskerDriver1 = {
        .name = "Whisker1",
        .init = WhiskerInit1,
        .test = WhiskerTest1,
        .usedPeriph = DECK_USING_UART2,
};

DECK_DRIVER(WhiskerDriver1);


/**
 * Logging variables for the Whisker
 */
LOG_GROUP_START(Whisker1)
LOG_ADD(LOG_FLOAT, Barometer2_1, &barometer2_1)
LOG_ADD(LOG_FLOAT, Barometer2_2, &barometer2_2)
LOG_ADD(LOG_FLOAT, Barometer2_3, &barometer2_3)
LOG_GROUP_STOP(Whisker1) 