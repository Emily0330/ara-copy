#ifdef SPIKE
#include "util.h"
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include <riscv_vector.h>
#include <string.h>

#define QK_I2_S 128
#define QK_I2 128
#define LUT_SIZE 16

__attribute__((aligned(32))) int8_t lut[LUT_SIZE] = {0,0,0,0,0,1,2,3,0,2,4,6,0,3,6,9};
void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;
    // for(int i=0;i<n;i++) printf("%4d", *((y + i))); // works normally
    // printf("\n");

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    
    // printf("group32_num: %4d\n", group32_num);
    
    // load LUT into AVX2 register and prepare it for 256-bit shuffle
    size_t vl = vsetvl_e8m1(32);  // initialize vl
    vint8m1_t lut_vec = vle8_v_i8m1(lut, vl);
    
    // // DEBUG: 印 lut_vec
    // int8_t lut_arr[16];
    // vse8_v_i8m1(lut_arr, lut_vec, vl);
    // printf("[DEBUG] lut_vec:\n");
    // for (int i = 0; i < 16; i++) printf("%4d ", lut_arr[i]);
    // printf("\n");

    lut_vec = vslideup_vx_i8m1(lut_vec, lut_vec, 16, vl); // 將低位 16 ~ vl - 1 個元素複製到高位

    // accumulator for final result
    vint32m4_t accu = vmv_v_x_i32m4(0, vl);
    vint32m4_t group_accu = vmv_v_x_i32m4(0, vl);

    printf("=== y raw ===\n"); // works normally
    for (int i = 0; i < n; i++) printf("%4d ", *(y+i));
    printf("\n");
    printf("la_num: %4d\n", la_num);

    for (int j = 0; j < la_num; j++) {
        printf("=== y raw ===\n"); // not working (print all zeroes)
        for (int i = 0; i < n; i++) printf("%4d ", *(y+i));
        printf("\n");

        const uint8_t *x_base = x + group32_num * 32 * 32 + j * 32;
        const int8_t *y_base = y + group32_num * 128 * 32 + j * 128;


        // printf("=== y_base raw ===\n"); // not working (print all zeroes)
        // for (int i = 0; i < n; i++) printf("%4d ", *(y_base+i));
        // printf("\n");

        // printf("=== x_base raw ===\n"); // works normally
        // for (int i = 0; i < 32; i++) printf("0x%0x ", x_base[i]);
        // printf("\n");


        // load activations and weights
        vint8m1_t act_vec_1 = vle8_v_i8m1(y_base + 0, vl);
        vint8m1_t act_vec_2 = vle8_v_i8m1(y_base + 32, vl);
        vint8m1_t act_vec_3 = vle8_v_i8m1(y_base + 64, vl);
        vint8m1_t act_vec_4 = vle8_v_i8m1(y_base + 96, vl);
        vuint8m1_t wt_vec_4  = vle8_v_u8m1(x_base, vl); // vint8m1_t->vuint8m1_t
        vuint8m1_t wt_vec_3 = vsrl_vx_u8m1(wt_vec_4, 2, vl); // vint8m1_t->vuint8m1_t
        vuint8m1_t wt_vec_2 = vsrl_vx_u8m1(wt_vec_4, 4, vl); // vint8m1_t->vuint8m1_t
        vuint8m1_t wt_vec_1 = vsrl_vx_u8m1(wt_vec_4, 6, vl); // vint8m1_t->vuint8m1_t

        vuint8m1_t wt_index_1 = vand_vv_u8m1(wt_vec_1, vmv_v_x_u8m1(0x03, vl), vl); // 32 weights
        vuint8m1_t wt_index_2 = vand_vv_u8m1(wt_vec_2, vmv_v_x_u8m1(0x03, vl), vl); // 32 weights
        vuint8m1_t wt_index_3 = vand_vv_u8m1(wt_vec_3, vmv_v_x_u8m1(0x03, vl), vl); // 32 weights
        vuint8m1_t wt_index_4 = vand_vv_u8m1(wt_vec_4, vmv_v_x_u8m1(0x03, vl), vl); // 32 weights

        // accumulator for partial results of this block
        vint16m2_t block_sub_result = vmv_v_x_i16m2(0, vl);

        // compute using LUT for all 4 shifts (2-bit processing)
        int shifts[5] = {0, 2, 4, 6, 7};
        for (int i = 0; i < 5; i++) {
            int shift = shifts[i];

            vuint8m1_t act_index_1, act_index_2, act_index_3, act_index_4;

            if (shift == 7 || shift == 6) {
                act_index_1 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_1), shift, vl), vmv_v_x_u8m1(0x01, vl), vl);
                act_index_2 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_2), shift, vl), vmv_v_x_u8m1(0x01, vl), vl);
                act_index_3 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_3), shift, vl), vmv_v_x_u8m1(0x01, vl), vl);
                act_index_4 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_4), shift, vl), vmv_v_x_u8m1(0x01, vl), vl);
            } else {
                act_index_1 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_1), shift, vl), vmv_v_x_u8m1(0x03, vl), vl);
                act_index_2 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_2), shift, vl), vmv_v_x_u8m1(0x03, vl), vl);
                act_index_3 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_3), shift, vl), vmv_v_x_u8m1(0x03, vl), vl);
                act_index_4 = vand_vv_u8m1(vsrl_vx_u8m1(vreinterpret_v_i8m1_u8m1(act_vec_4), shift, vl), vmv_v_x_u8m1(0x03, vl), vl);
            }

            // // DEBUG: 印 act_vec_1
            int8_t act_arr[32];
            // vse8_v_i8m1(act_arr, act_vec_1, vl);
            // printf("[DEBUG] act_vec_1:\n");
            // for (int i = 0; i < 32; i++) printf("%4d ", act_arr[i]);
            // printf("\n");

            // // DEBUG: 印 wt_vec_4
            uint8_t wt_arr[32];
            // vse8_v_u8m1(wt_arr, wt_vec_4, vl);
            // printf("[DEBUG] wt_vec_4 (raw packed 2-bit):\n");
            // for (int i = 0; i < 32; i++) printf("0x%x ", wt_arr[i]);
            // printf("\n");

            // build up indices
            vuint8m1_t combined_index_1 = vor_vv_u8m1(act_index_1, vsll_vx_u8m1(wt_index_1, 2, vl), vl);
            vuint8m1_t combined_index_2 = vor_vv_u8m1(act_index_2, vsll_vx_u8m1(wt_index_2, 2, vl), vl);
            vuint8m1_t combined_index_3 = vor_vv_u8m1(act_index_3, vsll_vx_u8m1(wt_index_3, 2, vl), vl);
            vuint8m1_t combined_index_4 = vor_vv_u8m1(act_index_4, vsll_vx_u8m1(wt_index_4, 2, vl), vl);

            // LUT lookup, replace shuffle in AVX2 with gather in RVV
            vint8m1_t lut_values_1 = vrgather_vv_i8m1(lut_vec, combined_index_1, vl);
            vint8m1_t lut_values_2 = vrgather_vv_i8m1(lut_vec, combined_index_2, vl);
            vint8m1_t lut_values_3 = vrgather_vv_i8m1(lut_vec, combined_index_3, vl);
            vint8m1_t lut_values_4 = vrgather_vv_i8m1(lut_vec, combined_index_4, vl);

            // // DEBUG: 印 combined_index_1 與對應的 lut values
            // vse8_v_u8m1(wt_arr, combined_index_1, vl);
            // printf("[DEBUG] combined_index_1:\n");
            // for (int i = 0; i < 32; i++) printf("%4d ", wt_arr[i]);
            // printf("\n");

            // vse8_v_i8m1(act_arr, lut_values_1, vl);
            // printf("[DEBUG] lut_values_1:\n");
            // for (int i = 0; i < 32; i++) printf("%4d ", act_arr[i]);
            // printf("\n");

            // extend from 8-bit to 16-bit
            vint16m2_t lut_values_low_16_1 = vwcvt_x_x_v_i16m2(lut_values_1, vl);
            vint16m2_t lut_values_low_16_2 = vwcvt_x_x_v_i16m2(lut_values_2, vl);
            vint16m2_t lut_values_low_16_3 = vwcvt_x_x_v_i16m2(lut_values_3, vl);
            vint16m2_t lut_values_low_16_4 = vwcvt_x_x_v_i16m2(lut_values_4, vl);

            if (shift == 7) {
                lut_values_low_16_1 = vsub_vv_i16m2(vmv_v_x_i16m2(0, vl), lut_values_low_16_1, vl);
                lut_values_low_16_2 = vsub_vv_i16m2(vmv_v_x_i16m2(0, vl), lut_values_low_16_2, vl);
                lut_values_low_16_3 = vsub_vv_i16m2(vmv_v_x_i16m2(0, vl), lut_values_low_16_3, vl);
                lut_values_low_16_4 = vsub_vv_i16m2(vmv_v_x_i16m2(0, vl), lut_values_low_16_4, vl);
            }

            // 累加 2-bit * 2-bit 查表結果
            block_sub_result = vadd_vv_i16m2(block_sub_result, vsll_vx_i16m2(vadd_vv_i16m2(lut_values_low_16_1, lut_values_low_16_2, vl), shift, vl), vl);
            block_sub_result = vadd_vv_i16m2(block_sub_result, vsll_vx_i16m2(vadd_vv_i16m2(lut_values_low_16_3, lut_values_low_16_4, vl), shift, vl), vl);

            // // === DEBUG 印出 block_sub_result ===
            // int16_t block_debug[64]; // 通常 32～64 足夠
            // vse16_v_i16m2(block_debug, block_sub_result, vl);
            // printf("[DEBUG] block_sub_result after shift=%d:\n", shift);
            // for (int i = 0; i < 32; i++) {
            //     printf("%6d ", block_debug[i]);
            //     if ((i + 1) % 8 == 0) printf("\n");
            // }
        }

        // 轉換低位到 32-bit 並累加最終結果
        vint32m4_t block_sub_result_32 = vwcvt_x_x_v_i32m4(block_sub_result, vl);
        group_accu = vadd_vv_i32m4(group_accu, block_sub_result_32, vl);
    }

    // accumulate group results into global accumulator
    accu = vadd_vv_i32m4(accu, group_accu, vl); 

    // final horizontal sum of the accumulator (sum of 32 32-bit numbers)
    int32_t sumi = 0;
    
    // sumi = vredsum_vs_i32m4_i32m1(sumi, accu, vsetvl_e32m4(32),vl);
    // 取得 vector length
    vl = vsetvl_e32m4(32);

    vint32m1_t zero = vmv_v_x_i32m1(0, vl); // scalar init
    vint32m1_t dst  = vmv_v_x_i32m1(0, vl); // output location

    vint32m1_t sumv = vredsum_vs_i32m4_i32m1(dst, accu, zero, vl);

    // === DEBUG: 印出 sumv 向量內容 ===
    // int32_t debug_sumv[4];  // vlen = 1~4 for m1
    // vse32_v_i32m1(debug_sumv, sumv, vsetvl_e32m1(1));
    // printf("[DEBUG] sumv (after vredsum):\n");
    // for (int i = 0; i < 1; i++) {
    //     printf("sumv[%d] = %d\n", i, debug_sumv[i]);
    // }

    // 把結果提取出來（int）
    sumi = vmv_x_s_i32m1_i32(sumv);
    // printf("sumi: %x\n", sumi);

    *s = (float)sumi; // C does not have static_cast
    // printf("*s: %x\n", *s);
}

int main() {
    const int n = 128;
    float result = 0.0f;
    float test = 5.0;
    if(test == 5) printf("test is 5\n");
    else printf("test is not 5\n");
    printf("test: %x\n",test);
    
    // 模擬 std::vector<uint8_t> vx
    uint8_t vx[128 / 4] = {
        0b10101010, 0b10101010, 0b11111111, 0b10101010,
        0b10101010, 0b10101010, 0b10101010, 0b10101010,
        0b10101010, 0b10101010, 0b10101010, 0b10101010,
        0b10101010, 0b10101010, 0b10101010, 0b10101010,
        0b10101010, 0b10101010, 0b11111111, 0b10101010,
        0b10101010, 0b10101010, 0b10101010, 0b10101010,
        0b10101010, 0b10101010, 0b10101010, 0b10101010,
        0b10101010, 0b10101010, 0b10101010, 0b10101010,
    };

    // 模擬 std::vector<int8_t> vy
    int8_t vy[128] = {0};
    for (int i = 0; i < 128; i += 16) {
        vy[i + 2] = -128;
    }

    // // 印出 vx（packed 2-bit）位元內容
    // printf("Weights (vx, packed 2-bit):\n");
    // for (int i = 0; i < sizeof(vx); i++) {
    //     printf("%x ", vx[i]);
    // }
    // printf("\n");

    // // 印出 vy（8-bit activations）
    // printf("Activations (vy, 8-bit):\n");
    // for (int i = 0; i < 128; i++) {
    //     printf("%4d ", vy[i]);
    //     if ((i + 1) % 16 == 0) printf("\n");
    // }

    // 執行 dot product
    ggml_vec_dot_i2_i8_s(n, &result, 0, vx, 0, vy, 0, 0);
    // ggml_vec_dot_i2_i8_s(n, &ans, 0, vx, 0, vy, 0, 0);

    // 輸出結果
    int32_t ans = -3072;
    if(result == ans) printf("The result is CORRECT!\n");
    else printf("The result is WRONG!\n");
    printf("Printed Result: %x\n", result); // Spike is unable to print float, which is a known issue
    printf("Actual Dot product result: %4d\n", ans);

    return 0;
}