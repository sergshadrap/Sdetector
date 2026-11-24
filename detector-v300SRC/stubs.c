#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#ifdef __cplusplus
extern "C" {
#endif
int memcpy_s(void*d,size_t dm,const void*s,size_t n){ if(n>dm) return 34; memcpy(d,s,n); return 0; }
int memmove_s(void*d,size_t dm,const void*s,size_t n){ if(n>dm) return 34; memmove(d,s,n); return 0; }
int memset_s(void*d,size_t dm,int c,size_t n){ if(n>dm) return 34; memset(d,c,n); return 0; }
int strncpy_s(char*d,size_t dm,const char*s,size_t n){ if(!dm) return 34; size_t k=n<dm?n:dm-1; strncpy(d,s,k); d[k]=0; return 0; }
int snprintf_s(char*d,size_t dm,size_t n,const char*f,...){ va_list a; va_start(a,f); int r=vsnprintf(d,dm,f,a); va_end(a); return r; }
#define S(x) int x(){ return 0; }
S(AUDIO_UPVQE_Destroy) S(AUDIO_UPVQE_GetConfig) S(AUDIO_UPVQE_WriteFrame) S(AUDIO_UPVQE_ReadFrame)
S(AUDIO_UPVQE_Create) S(AUDIO_UPVQE_GetVolume) S(AUDIO_UPVQE_SetVolume)
S(VOICE_EncodeFrame) S(VOICE_DecodeFrame) S(VOICE_DecReset) S(VOICE_EncReset)
S(AUDIO_DNVQE_Destroy) S(AUDIO_DNVQE_ReadFrame) S(AUDIO_DNVQE_GetConfig)
S(AUDIO_DNVQE_Create) S(AUDIO_DNVQE_WriteFrame) S(AUDIO_VQE_RegisterModuleHandle)
#ifdef __cplusplus
}
#endif