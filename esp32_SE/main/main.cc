/*
NOTE: All of this is for running tests and should be cleaned up
*/
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "esp_attr.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "gtcrn_micro_int8_data.cc"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TFLITE_SCHEMA_VERSION (3)
// logger for esp32
static const char *TAG = "GTCRN_MICRO_TEST";

// Settings
static const gpio_num_t led_pin = GPIO_NUM_38;
static const int32_t sleep_time_ms = 200;

// allocate Tensor Arena size on PSRAM
constexpr int kTensorArenaSize = 7500 * 1024;
uint8_t *tensor_arena = nullptr;

// logging
void error_blink(const char *msg) {
  while (1) {
    printf("ERROR: %s\n", msg);
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

extern "C" void app_main(void) {

  ESP_LOGI(TAG, "Starting gtcrn_micro test");
  ESP_LOGI(TAG, "Free internal heap: %d", (int)esp_get_free_heap_size());
  ESP_LOGI(TAG, "Free SPIRAM: %d",
           (int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // heap allocate tensor arena for debugging
  tensor_arena = (uint8_t *)heap_caps_malloc(
      kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  // debug malloc
  if (!tensor_arena) {
    ESP_LOGE(TAG, "Failed to allocate Tensor in PSRAM");
    error_blink("PSRAM Arena malloc failed");
    return;
  }

  // worked?
  ESP_LOGI(TAG, "Arena allocated at %p, size %d", tensor_arena,
           kTensorArenaSize);

  // log remaining memory
  ESP_LOGI(TAG, "Remaining internal heap %d", (int)esp_get_free_heap_size());
  ESP_LOGI(TAG, "Remaining SPIRAM: %d",
           (int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // SETTING UP TF
  const tflite::Model *model =
      tflite::GetModel(gtcrn_micro_full_integer_quant_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema %" PRIu32 "not equal to supported %" PRIu32,
             static_cast<u_int32_t>(model->version()),
             static_cast<u_int32_t>(TFLITE_SCHEMA_VERSION));
    error_blink("Schema mismatch");
    return;
  }

  // resolve ops
  // static tflite::AllOpsResolver resolver;
  static tflite::MicroMutableOpResolver<24> resolver;
  // registering the ops
  resolver.AddGather();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddTranspose();
  resolver.AddReshape();
  resolver.AddDequantize();
  resolver.AddSqrt();
  resolver.AddQuantize();
  resolver.AddConcatenation();
  resolver.AddStridedSlice();
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddAbs();
  resolver.AddSub();
  resolver.AddRelu();
  resolver.AddPad();
  resolver.AddDepthwiseConv2D();
  resolver.AddTransposeConv();
  resolver.AddTanh();

  // setup interpreter
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                              kTensorArenaSize);

  // checking memory tensor allocation
  TfLiteStatus alloc_status = interpreter.AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed");
    error_blink("Alloc failed");
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
      data[i] = 0;
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

  // Invoke once
  // do a timer check
  ESP_LOGI(TAG, "Invoking model...");
  int64_t t0 = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter.Invoke();
  int64_t t1 = esp_timer_get_time();

  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke() failed");
    return;
  } else {
    ESP_LOGI(TAG, "Invoke time: %lld us", (long long)(t1 - t0));
  }

  ESP_LOGI(TAG, "Inference succeeded – blinking LED");

  int8_t led_state = 0;
  // config GPIO
  gpio_reset_pin(led_pin);
  gpio_set_direction(led_pin, GPIO_MODE_OUTPUT);
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
