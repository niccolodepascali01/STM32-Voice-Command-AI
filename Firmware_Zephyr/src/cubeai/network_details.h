/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-06-09T15:33:44+0000
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[13]) {
   { .size_bytes = 63984, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 124, 129, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "serving_default_keras_tensor_220_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 32, 32, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "resize_0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 32, 32, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_1_output" },
   { .size_bytes = 65536, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 32, 32, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_2_output" },
   { .size_bytes = 65536, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 32, 32, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_2_nl_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 16, 16, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pool_3_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 16, 16, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_4_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 16, 16, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_4_nl_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 8, 8, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pool_5_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_10_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_10_nl_output" },
   { .size_bytes = 20, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 5}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_11_output" },
   { .size_bytes = 20, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 5}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_12_output" }
  },
  .nodes = (const stai_node_details[12]){
    {.id = 0, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* resize_0 */
    {.id = 1, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* eltwise_1 */
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_2 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* nl_2_nl */
    {.id = 3, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* pool_3 */
    {.id = 4, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* conv2d_4 */
    {.id = 4, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* nl_4_nl */
    {.id = 5, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* pool_5 */
    {.id = 10, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* gemm_10 */
    {.id = 10, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* nl_10_nl */
    {.id = 11, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* gemm_11 */
    {.id = 12, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} } /* nl_12 */
  },
  .n_nodes = 12
};
#endif

