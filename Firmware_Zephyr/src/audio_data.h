#ifndef AUDIO_DATA_H
#define AUDIO_DATA_H

#include <stdint.h>

// Mettiamo il "PROGMEM" (o l'equivalente per STM32) dichiarandolo extern
extern const int16_t raw_audio_yes[16000];
extern const int16_t raw_audio_no[16000];
extern const int16_t raw_audio_stop[16000];
extern const int16_t raw_audio_go[16000];
extern const int16_t raw_audio_unknown[16000];

#endif