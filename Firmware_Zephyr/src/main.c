#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <stdio.h>
#include <string.h>
#include <arm_math.h>
#include "audio_data.h"

#include "stai.h"
#include "network.h"
#include "network_data.h"

/* --- PARAMETRI DSP --- */
#define FFT_SIZE     256
#define NUM_BINS     129
#define FRAME_LENGTH 255
#define FRAME_STEP   128
#define NUM_FRAMES   124

/* ===================================================
 * BUFFER IA API 4.0
 * =================================================== */
STAI_NETWORK_CONTEXT_DECLARE(network_ctx, STAI_NETWORK_CONTEXT_SIZE)

static arm_rfft_fast_instance_f32 fft_instance;
static float hann_window[FRAME_LENGTH];
static float frame_tmp[FFT_SIZE];
static float output_fft[FFT_SIZE];
static float output_fft_mag[NUM_BINS];

/* IL MOTORE: La memoria RAM centrale (70 KB) */
static uint8_t activations[STAI_NETWORK_ACTIVATIONS_SIZE_BYTES] __attribute__((aligned(4)));

static float *in_data = NULL;
static float *out_data = NULL;

const char *words[] = {"yes", "no", "stop", "go", "unknown"};

/* ===================================================
 * GESTIONE AUDIO DINAMICO E LED
 * =================================================== */
// Raccogliamo tutti i puntatori agli array in una lista (salvati in Flash)
const int16_t *audio_playlist[] = {
    raw_audio_yes, 
    raw_audio_no, 
    raw_audio_stop, 
    raw_audio_go, 
    raw_audio_unknown
};

// Variabile controllata dal pulsante per scegliere quale array passare alla rete
static volatile int current_word_idx = 0;

// Variabile aggiornata dall'IA per dire al LED che pattern eseguire (-1 = in attesa)
static volatile int predicted_pattern = -1;


void stft(int16_t *current_audio){
     for (int f = 0; f < NUM_FRAMES; f++) {
            int start = f * FRAME_STEP;
            for (int i = 0; i < FRAME_LENGTH; i++) {
                frame_tmp[i] = ((float)current_audio[start + i] / 32768.0f) * hann_window[i];
            }
            frame_tmp[255] = 0.0f;
            arm_rfft_fast_f32(&fft_instance, frame_tmp, output_fft, 0);
            output_fft_mag[0]   = fabsf(output_fft[0]);
            output_fft_mag[128] = fabsf(output_fft[1]);
            arm_cmplx_mag_f32(&output_fft[2], &output_fft_mag[1], 127);
            
            memcpy(&in_data[f * NUM_BINS], output_fft_mag, NUM_BINS * sizeof(float));
        }
}



/* ===================================================
 * 1. CONFIGURAZIONE HARDWARE (DEVICE TREE)
 * =================================================== */
#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

#define SW0_NODE DT_ALIAS(sw0)
static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(SW0_NODE, gpios);

/* ===================================================
 * 2. THREAD INTELLIGENZA ARTIFICIALE + DSP
 * =================================================== */
void ai_thread_entry(void) {
    printf("[1] Avvio...\r\n");

    if (stai_runtime_init() != STAI_SUCCESS) return;
    for (int n = 0; n < FRAME_LENGTH; n++) {
        hann_window[n] = 0.5f - 0.5f * arm_cos_f32(2.0f * PI * n / 255.0f);
    }
    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);
    if (stai_network_init(network_ctx) != STAI_SUCCESS) return;

    /* Setup Memoria Ottimizzata */
    stai_ptr act_ptr_arr[] = { (stai_ptr)activations };
    stai_network_set_activations(network_ctx, act_ptr_arr, 1);

    stai_ptr raw_in = NULL;
    stai_size n_buf = 0;
    stai_network_get_inputs(network_ctx, &raw_in, &n_buf);
    in_data = (float *)raw_in;

    stai_ptr raw_out = NULL;
    stai_network_get_outputs(network_ctx, &raw_out, &n_buf);
    out_data = (float *)raw_out;

    printf("[8] Pronto, avvio loop di Inferenza\r\n");

    while (1) {
        memset(in_data, 0, STAI_NETWORK_IN_1_SIZE_BYTES);

        // Catturiamo il puntatore all'array attualmente selezionato dal bottone
        const int16_t *current_audio = audio_playlist[current_word_idx];
        
        printf("\r\n========================================\r\n");
        printf("🎙️  ANALIZZANDO L'AUDIO: '%s'\r\n", words[current_word_idx]);
        printf("========================================\r\n");



        stft(current_audio);
       
        if (stai_network_run(network_ctx, STAI_MODE_SYNC) == STAI_SUCCESS) {
            int best_idx = 0;
            float max_prob = 0.0f;
            
            for (int idx = 0; idx < STAI_NETWORK_OUT_1_SIZE; idx++) {
                float prob = out_data[idx];
                printf("%-7s: %.4f\r\n", words[idx], (double)prob);
                if (prob > max_prob) { max_prob = prob; best_idx = idx; }
            }
            printf("----------------------------------------\r\n");
            printf("PREDIZIONE: -> %s <- (%.2f%%)\r\n", words[best_idx], (double)(max_prob * 100.0f));
            
            // Diciamo al LED quale coreografia eseguire
            predicted_pattern = best_idx;
        }

        k_msleep(2000); // Pausa di 2 secondi tra una lettura e l'altra
    }
}

/* ===================================================
 * 3. THREAD LED (Pattern Visivi)
 * =================================================== */
void led_thread_entry(void) {
    if (!gpio_is_ready_dt(&led)) return;
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
    
    while (1) {
        int pattern = predicted_pattern;
        
        if (pattern == 4) {
            // UNKNOWN: Lampeggio frenetico "Errore/Sconosciuto"
            gpio_pin_set_dt(&led, 1);
            k_msleep(50);
            gpio_pin_set_dt(&led, 0);
            k_msleep(50);
        } 
        else if (pattern >= 0 && pattern <= 3) {
            // YES(1), NO(2), STOP(3), GO(4) lampeggi seguiti da una pausa
            int blinks = pattern + 1;
            for(int i = 0; i < blinks; i++) {
                gpio_pin_set_dt(&led, 1);
                k_msleep(150); // Led ON
                gpio_pin_set_dt(&led, 0);
                k_msleep(150); // Led OFF
            }
            k_msleep(1000); // Pausa lunga prima di ripetere il pattern
        } 
        else {
            // In attesa della prima inferenza (Led spento)
            gpio_pin_set_dt(&led, 0);
            k_msleep(100);
        }
    }
}

/* ===================================================
 * 4. THREAD BOTTONE (Cambia Parola)
 * =================================================== */
void button_thread_entry(void) {
    if (!gpio_is_ready_dt(&button)) return;
    gpio_pin_configure_dt(&button, GPIO_INPUT);
    
    while (1) {
        if (gpio_pin_get_dt(&button) == 1) {
            // Passa alla parola successiva (0->1->2->3->4->0...)
            current_word_idx = (current_word_idx + 1) % 5;
            
            printf("\r\n****************************************\r\n");
            printf("🔘 PULSANTE PREMUTO! Passo all'audio: %s\r\n", words[current_word_idx]);
            printf("****************************************\r\n");
            
            // Spegniamo il pattern LED finché la rete non calcola il nuovo risultato
            predicted_pattern = -1; 
            
            k_msleep(300); // Debounce per evitare pressioni multiple
        }
        k_msleep(50);
    }
}

K_THREAD_DEFINE(ai_tid,  8192, ai_thread_entry,     NULL, NULL, NULL, 5, 0, 0);
K_THREAD_DEFINE(led_tid, 1024, led_thread_entry,    NULL, NULL, NULL, 5, 0, 0);
K_THREAD_DEFINE(btn_tid, 1024, button_thread_entry, NULL, NULL, NULL, 5, 0, 0);