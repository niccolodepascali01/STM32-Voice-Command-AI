[README.md](https://github.com/user-attachments/files/28791201/README.md)
# STM32 Voice Command AI — Keyword Spotting (KWS)

Progetto universitario di **TinyML per il riconoscimento di comandi vocali su microcontrollore STM32**, realizzato come esame del corso di sistemi embedded. Il sistema classifica quattro parole chiave (*yes*, *no*, *stop*, *go*) più una classe *unknown*, usando una rete neurale convoluzionale addestrata in TensorFlow e distribuita tramite **ST Edge AI Core** su una scheda **STM32 Nucleo** con **Zephyr RTOS**.

---

## Panoramica del sistema

Il progetto è diviso in due parti indipendenti ma strettamente collegate:

1. **Pipeline ML** (Python/Colab) — addestramento del modello, esportazione in TFLite, verifica.
2. **Firmware Zephyr** (C, STM32F4) — preprocessing DSP on-device, inferenza in tempo reale, output su LED e pulsante.

Il punto critico del progetto è che il **preprocessing audio nel firmware deve replicare esattamente** quello usato in training. Se i parametri STFT non coincidono, il modello riceve feature completamente diverse da quelle su cui è stato addestrato e le predizioni diventano casuali.

```
Audio PCM grezzo (16 kHz, int16, 1 secondo)
            │
            ▼
   Normalizzazione float32  (÷ 32768)
            │
            ▼
   Finestra di Hann  (255 campioni)
            │
            ▼
   Zero-padding → 256 campioni
            │
            ▼
   arm_rfft_fast_f32  (FFT reale a 256 punti)  ← CMSIS-DSP
            │
            ▼
   Spettro di magnitudine  (129 bin in frequenza)
            │
            ▼
   Ripetuto per 124 frame sovrapposti  (step = 128)
            │
            ▼
   Spettrogramma  [124 × 129 × 1]  →  INPUT RETE
            │
            ▼
   Inferenza CNN  (STAI API 4.x, Cortex-M4 FPU)
            │
            ▼
   Softmax → 5 classi  →  Pattern LED
```

---

## Struttura del repository

```
STM32-Voice-Command-AI/
│
├── Machine_Learning/
│   ├── Rete_riconoscimento_parole_migliorata.ipynb   # Pipeline di training completa
│   ├── model_kws.tflite                              # Modello addestrato ed esportato (~44 KB)
│   ├── raw_audio_yes.txt                             # Array PCM int16 in formato C (per il test finale)
│   ├── raw_audio_no.txt
│   ├── raw_audio_stop.txt
│   ├── raw_audio_go.txt
│   └── raw_audio_unknown.txt
│
├── Firmware_Zephyr/
│   ├── CMakeLists.txt                                # Build system CMake per Zephyr
│   ├── prj.conf                                      # Configurazione Kconfig (FPU, CMSIS-DSP, heap...)
│   ├── lib/
│   │   └── NetworkRuntime1201_CM4_GCC.a              # Runtime STAI statico precompilato per Cortex-M4
│   └── src/
│       ├── main.c                                    # Tre thread Zephyr: AI+DSP, LED, pulsante
│       ├── audio_data.c                              # Array audio costanti memorizzati in Flash
│       ├── audio_data.h
│       └── cubeai/                                   # Codice C della rete generato da ST Edge AI Core
│           ├── network.c / network.h                 # Grafo della rete (architettura + forward pass)
│           ├── network_data.c / network_data.h       # Pesi della rete (138 KiB in Flash)
│           ├── network_details.h                     # Dettagli tensor per tensor
│           ├── network_generate_report.txt           # Report completo di generazione (MACCs, RAM, Flash)
│           └── Inc/                                  # Header del runtime STAI (stai.h, layers, ecc.)
│
└── Utils/
    └── Script_python_audio.ipynb                     # Utility: ascolta un array PCM grezzo come WAV
```

---

## Pipeline di Machine Learning

### Dataset

Il training utilizza il dataset **Google Speech Commands (mini)**, scaricato automaticamente dal notebook.

- **Classi target (4):** *yes*, *no*, *stop*, *go*
- **Classe unknown:** costruita da campioni di *up*, *down*, *left*, *right* (bilanciata per numero rispetto alle classi target)
- **Formato:** file WAV mono, 16 kHz, 16-bit, ~1 secondo

La classe *unknown* non è nel dataset originale come etichetta, ma viene costruita artificialmente per insegnare al modello a rifiutare parole non riconosciute, evitando classificazioni forzate su qualcosa di non supportato.

### Preprocessing

Ogni campione WAV viene portato a esattamente **16 000 campioni** (zero-padding o troncamento), poi si calcola lo Short-Time Fourier Transform (STFT):

| Parametro        | Valore |
|-----------------|--------|
| `frame_length`  | 255 campioni |
| `frame_step`    | 128 campioni |
| Punti FFT       | 256 (implicito da TensorFlow) |
| Bin in frequenza| 129 (`frame_length // 2 + 1`) |
| Frame temporali | 124 |
| Shape output    | **(124, 129, 1)** — float32 |

Viene presa la **magnitudine** dello spettrogramma complesso (`tf.abs`). Un layer `Normalization` (adattato sul training set) standardizza l'input a media zero e varianza unitaria prima della rete.

### Architettura del modello

```
Input(124, 129, 1)
  └─ Resizing(32, 32)            ← riduzione risoluzione per abbassare i parametri
  └─ Normalization                ← media e varianza calcolate sul training set
  └─ Conv2D(16 filtri, 3×3, relu, padding=same)
  └─ MaxPooling2D(2×2)            → (16, 16, 16)
  └─ Conv2D(16 filtri, 3×3, relu, padding=same)
  └─ MaxPooling2D(2×2)            → (8, 8, 16)
  └─ Dropout(0.3)
  └─ Flatten                      → 1024
  └─ Dense(32, relu)
  └─ Dense(5, softmax)            → probabilità per 5 classi
```

#### Distribuzione dei MACCs per layer (sul Cortex-M4)

| Layer          | MACCs     | % totale | Note |
|----------------|-----------|----------|------|
| `conv2d_2`     | 163 856   | 20.1 %   | Prima convoluzione + ReLU |
| `conv2d_4`     | 593 936   | 72.7 %   | Seconda conv — **layer dominante** |
| `pool_3`       | 16 384    |  2.0 %   | MaxPool |
| `gemm_10`      | 32 832    |  4.0 %   | Fully Connected 1024→32 |
| `gemm_11`      | 165       |  0.0 %   | Fully Connected 32→5 |
| `nl_12`        | 75        |  0.0 %   | Softmax finale |
| Altro          | ~9 216    |  1.1 %   | Resize, sub, pool_5 |
| **Totale**     | **816 464**| **100%** | |

#### Occupazione di memoria (target STM32F4)

| Risorsa              | Dimensione   |
|----------------------|-------------|
| Pesi (Flash/rodata)  | 138.46 KiB  |
| Attivazioni (RAM)    | 68.04 KiB   |
| Runtime STAI (Flash) | ~10 KiB     |
| **FLASH totale**     | **~153 KiB**|
| **RAM totale**       | **~70 KiB** |
| Parametri trainabili | 35 446      |
| Dimensione TFLite    | ~44 KB      |

### Addestramento

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)
```

Il notebook include anche la matrice di confusione, l'accuracy per classe e i grafici di loss/accuracy durante il training.

### Esportazione

Il modello viene esportato in TFLite con ottimizzazione di default (post-training quantization):

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model_kws.tflite', 'wb') as f:
    f.write(tflite_model)
```

Il file `.tflite` risultante viene poi passato a **ST Edge AI Core** per la generazione del codice C da integrare nel firmware.

### Generazione del codice C (ST Edge AI Core)

Il codice nella cartella `cubeai/` è generato automaticamente da ST Edge AI Core v4.0.1 a partire da `model_kws.tflite`:

```bash
stm32ai generate \
  --model model_kws.tflite \
  --target stm32f4 \
  --optimization balanced \
  --name network
```

I file generati (`network.c`, `network_data.c`, `network.h`, `network_data.h`, `network_details.h`) non vanno modificati a mano — sono da rigenerare ogni volta che il modello cambia.

### Dipendenze Python

```
tensorflow >= 2.10
numpy
scipy
matplotlib
seaborn
scikit-learn
```

---

## Firmware Zephyr

### Hardware target

- **Scheda:** STM32 Nucleo (serie STM32F4, Cortex-M4 con FPU hardware)
- **RTOS:** [Zephyr RTOS](https://docs.zephyrproject.org/)
- **AI runtime:** ST Edge AI Core v4.0.1 — STAI API 4.x (lite runtime)
- **DSP:** CMSIS-DSP (libreria ARM ottimizzata per Cortex-M)

### Configurazione Zephyr (`prj.conf`)

```conf
# Memoria
CONFIG_MAIN_STACK_SIZE=4096
CONFIG_HEAP_MEM_POOL_SIZE=8192   # L'API STAI 4.x può fare allocazioni interne durante init

# C library e console
CONFIG_PICOLIBC=y
CONFIG_PRINTK=y
CONFIG_STDOUT_CONSOLE=y
CONFIG_CBPRINTF_FP_SUPPORT=y     # Supporto printf per float

# Hardware
CONFIG_FPU=y
CONFIG_FP_HARDABI=y              # ABI hardware FPU (obbligatorio per CMSIS-DSP e STAI)
CONFIG_GPIO=y

# CMSIS-DSP
CONFIG_CMSIS_DSP=y
CONFIG_CMSIS_DSP_TRANSFORM=y     # arm_rfft_fast_f32
CONFIG_CMSIS_DSP_COMPLEXMATH=y   # arm_cmplx_mag_f32
CONFIG_CMSIS_DSP_FASTMATH=y      # arm_cos_f32 (per la finestra di Hann)
```

### Architettura multi-thread

Il firmware usa **tre thread Zephyr** indipendenti che comunicano tramite variabili `volatile` condivise:

```
┌─────────────────────────────────────────────────────────────────┐
│  ai_thread   (stack 8 KiB, priorità 5)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Init STAI runtime + FFT + finestra di Hann              │    │
│  │ Loop:                                                   │    │
│  │   legge current_word_idx ──────────────────────────┐   │    │
│  │   preprocessing DSP (Hann + rFFT × 124 frame)      │   │    │
│  │   stai_network_run()                                │   │    │
│  │   aggiorna predicted_pattern ◄──────────────────────┘   │    │
│  │   k_msleep(2000)                                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
        │  predicted_pattern (volatile int)  ▲
        ▼                                   │
┌─────────────────────────────────────────────────────────────────┐
│  led_thread  (stack 1 KiB, priorità 5)                          │
│  Pattern LED basato su predicted_pattern:                        │
│    yes=1 blink, no=2 blinks, stop=3 blinks, go=4 blinks         │
│    unknown=lampeggio rapido 50ms, -1=LED spento (in attesa)      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  button_thread  (stack 1 KiB, priorità 5)                        │
│  Polling del pulsante ogni 50ms (debounce 300ms)                 │
│  Incrementa current_word_idx (0→1→2→3→4→0...)                   │
│  Resetta predicted_pattern = -1 al cambio di parola             │
└─────────────────────────────────────────────────────────────────┘
```

### Preprocessing DSP nel firmware

Il codice C replica esattamente il comportamento di `tf.signal.stft` con gli stessi parametri:

```c
#define FFT_SIZE     256    // Punti FFT (potenza di 2 per arm_rfft_fast)
#define NUM_BINS     129    // FFT_SIZE/2 + 1 = bin in frequenza
#define FRAME_LENGTH 255    // Lunghezza del frame (come in Python)
#define FRAME_STEP   128    // Sovrapposizione tra frame
#define NUM_FRAMES   124    // Frame totali per 1 secondo a 16 kHz

// Pre-calcolo finestra di Hann (fatto una volta sola all'init)
for (int n = 0; n < FRAME_LENGTH; n++) {
    hann_window[n] = 0.5f - 0.5f * arm_cos_f32(2.0f * PI * n / 255.0f);
}

// Per ogni frame:
for (int f = 0; f < NUM_FRAMES; f++) {
    int start = f * FRAME_STEP;

    // 1. Normalizza int16 → float32 e applica la finestra di Hann
    for (int i = 0; i < FRAME_LENGTH; i++) {
        frame_tmp[i] = ((float)audio[start + i] / 32768.0f) * hann_window[i];
    }
    frame_tmp[255] = 0.0f;  // Zero-padding all'ultimo campione per arrivare a 256

    // 2. FFT reale a 256 punti (CMSIS-DSP)
    arm_rfft_fast_f32(&fft_instance, frame_tmp, output_fft, 0);

    // 3. Estrazione magnitudine (129 bin)
    output_fft_mag[0]   = fabsf(output_fft[0]);    // DC
    output_fft_mag[128] = fabsf(output_fft[1]);    // Nyquist
    arm_cmplx_mag_f32(&output_fft[2], &output_fft_mag[1], 127);  // Bin 1..127

    // 4. Copia nella riga f dello spettrogramma (input della rete)
    memcpy(&in_data[f * NUM_BINS], output_fft_mag, NUM_BINS * sizeof(float));
}
```

> **Nota:** la rete include internamente un layer `Normalization` che normalizza l'input. La media e la varianza vengono incorporate direttamente nei pesi durante la conversione TFLite → C, quindi non è necessario applicarle manualmente nel firmware.

### Campioni audio in Flash

Il firmware non usa un microfono: cinque campioni audio di test (uno per classe) sono salvati come array `const int16_t` in Flash e selezionati tramite pulsante.

```c
// audio_data.h
extern const int16_t raw_audio_yes[16000];
extern const int16_t raw_audio_no[16000];
extern const int16_t raw_audio_stop[16000];
extern const int16_t raw_audio_go[16000];
extern const int16_t raw_audio_unknown[16000];
```

Occupazione Flash per i campioni: 5 × 16 000 × 2 byte = **~160 KiB**.

I file `raw_audio_*.txt` nella cartella `Machine_Learning/` contengono gli stessi dati in formato testuale (array C) — generati dal notebook a partire dal validation set per verificare le predizioni anche su PC prima di flashare.

### Gestione della memoria per l'inferenza

Tutto il buffer di attivazioni della rete è allocato staticamente (nessun `malloc` a runtime):

```c
// Buffer da 68 KiB per le attivazioni (input, output e layer intermedi)
static uint8_t activations[STAI_NETWORK_ACTIVATIONS_SIZE_BYTES] __attribute__((aligned(4)));

// Setup una tantum all'avvio
stai_ptr act_ptr_arr[] = { (stai_ptr)activations };
stai_network_set_activations(network_ctx, act_ptr_arr, 1);
```

Input e output della rete vengono recuperati come puntatori al buffer di attivazioni (zero-copy):

```c
stai_network_get_inputs(network_ctx, &raw_in, &n_buf);
in_data = (float *)raw_in;   // puntatore diretto all'area input

stai_network_get_outputs(network_ctx, &raw_out, &n_buf);
out_data = (float *)raw_out; // puntatore diretto all'area output
```

### Codifica LED

| Predizione  | Pattern LED                       |
|-------------|-----------------------------------|
| *yes*       | 1 blink (150 ms on / 150 ms off)  |
| *no*        | 2 blinks                          |
| *stop*      | 3 blinks                          |
| *go*        | 4 blinks                          |
| *unknown*   | Lampeggio rapido (50 ms on/off)   |
| In attesa   | LED spento                        |

### Output seriale (UART)

A ogni inferenza il firmware stampa le probabilità per tutte le classi e la predizione vincente:

```
========================================
🎙️  ANALIZZANDO L'AUDIO: 'go'
========================================
yes    : 0.0031
no     : 0.0012
stop   : 0.0008
go     : 0.9921
unknown: 0.0028
----------------------------------------
PREDIZIONE: -> go <- (99.21%)
```

---

## Build e Flash

### Prerequisiti

- [Zephyr SDK](https://docs.zephyrproject.org/latest/develop/getting_started/index.html) installato e `ZEPHYR_BASE` impostato
- `west` disponibile nel PATH
- Scheda STM32 Nucleo collegata via USB (ST-Link integrato)

### Compilazione e flashing

```bash
# Dalla root del repository
west build -b nucleo_f401re Firmware_Zephyr
west flash
```

> Sostituire `nucleo_f401re` con il target corretto per la propria scheda (es. `nucleo_f446re`, `nucleo_f411re`).

### Monitor seriale

```bash
# Con west (se configurato)
west espresso

# Oppure con qualsiasi terminale seriale a 115200 baud
minicom -D /dev/ttyACM0 -b 115200
```

### Rigenerare il codice C dopo un nuovo training

Se si riaddestrare il modello e si ottiene un nuovo `model_kws.tflite`:

1. Aprire [ST Edge AI Developer Cloud](https://stm32ai.st.com/) (o usare il tool locale)
2. Caricare il `.tflite` con i parametri: `target = stm32f4`, `optimization = balanced`, `name = network`
3. Sostituire i file in `Firmware_Zephyr/src/cubeai/` con quelli generati
4. Ricompilare il firmware

---

## Utility

`Utils/Script_python_audio.ipynb` — permette di incollare un array PCM grezzo (uno dei `raw_audio_*.txt`) e ascoltarlo come file WAV nel browser. Utile per verificare visivamente e acusticamente che i campioni estratti dal validation set siano corretti prima di compilarli nel firmware.

---

## Scelte progettuali principali

**Audio in Flash invece che da microfono** — semplifica enormemente il firmware per un progetto dimostrativo: nessun driver audio, nessun problema di timing, campioni sempre identici e riproducibili.

**Preprocessing C che replica tf.signal.stft** — i parametri `frame_length=255`, `frame_step=128`, FFT a 256 punti e magnitudine su 129 bin sono identici in training e in inferenza. Qualsiasi differenza (es. usare una finestra rettangolare invece di Hann, o sbagliare il numero di frame) degrada drasticamente l'accuracy sul campo.

**STAI lite runtime con `use-lite-runtime` + `use-st-ai`** — il codice generato usa il runtime leggero di ST Edge AI, che occupa solo ~10 KiB di Flash aggiuntivi rispetto al modello.

**Assenza di allocazione dinamica a runtime** — attivazioni e buffer I/O sono statici e allineati a 4 byte. Questo garantisce comportamento deterministico e assenza di frammentazione heap su un sistema embedded.

**Tre thread Zephyr separati** — disaccoppia completamente la logica AI (lenta, ~2 s/ciclo) dall'output LED (continuo) e dalla lettura del pulsante (polling a 50 ms), senza bisogno di logica asincrona complessa.
