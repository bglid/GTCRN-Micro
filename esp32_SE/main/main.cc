#include <stdint.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "gtcrn_micro_int8_data.cc"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// logger for esp32
static const char *TAG = "GTCRN_MICRO_TEST";

// Settings
static const gpio_num_t led_pin = GPIO_NUM_4;
static const int32_t sleep_time_ms = 1000;

static constexpr int kTensorArenaSize = 400 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main(void) {
  int8_t led_state = 0;

  // config GPIO
  gpio_reset_pin(led_pin);
  gpio_set_direction(led_pin, GPIO_MODE_OUTPUT);

  ESP_LOGI(TAG, "Starting gtcrn_micro test");

  // SETTING UP TF
  const tflite::Model *model = tflite::GetModel(gtcrn_micro_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema %d not equal to supported %d", model->version(),
             TFLITE_SCHEMA_VERSION);
    return;
  }

  // resolve ops
  static tflite::AllOpsResolver resolver;

  // setup interpreter
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                              kTensorArenaSize);

  // checking memory tensor allocation
  TfLiteStatus alloc_status = interpreter.AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed");
    return;
  }

  TfLiteTensor *input = interpreter.input(0);
  ESP_LOGI(TAG, "Input tensor: type=%d, dims[0..3]=(%d,%d,%d,%d)", input->type,
           input->dims->data[0], input->dims->data[1], input->dims->data[2],
           input->dims->data[3]);

  // running with dummy input
  if (input->type == kTfLiteInt8) {
    int8_t *data = input->data.int8;
    int total = 1;
    for (int i = 0; i < input->dims->size; ++i) {
      total *= input->dims->data[i];
    }
    for (int i = 0; i < total; ++i) {
      data[i] = 0; // silence / zero STFT
    }
  } else if (input->type == kTfLiteFloat32) {
    float *data = input->data.f;
    int total = 1;
    for (int i = 0; i < input->dims->size; ++i) {
      total *= input->dims->data[i];
    }
    for (int i = 0; i < total; ++i) {
      data[i] = 0.0f;
    }
  } else {
    ESP_LOGE(TAG, "Unexpected input type: %d", input->type);
    return;
  }

  // 5) Invoke once
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke() failed");
    return;
  }

  ESP_LOGI(TAG, "Inference succeeded – blinking LED");

  // loop and blink LED if nothing else failed!
  while (1) {
    // toggle the LED
    led_state = !led_state;
    gpio_set_level(led_pin, led_state);

    // hello world!
    printf("LED State: %d\n", led_state);

    // freeRTOS delay
    vTaskDelay(sleep_time_ms / portTICK_PERIOD_MS);
  }
}
