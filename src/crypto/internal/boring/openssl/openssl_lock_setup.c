// +build linux
// +build openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

#include "goopenssl.h"
#include <stdio.h>
#include <pthread.h>
#include <openssl/err.h>
#include <sys/types.h>
#include <sys/syscall.h>

#define _GNU_SOURCE
#include <unistd.h>
 
#define MUTEX_TYPE       pthread_mutex_t
#define MUTEX_SETUP(x)   pthread_mutex_init(&(x), NULL)
#define MUTEX_CLEANUP(x) pthread_mutex_destroy(&(x))
#define MUTEX_LOCK(x)    pthread_mutex_lock(&(x))
#define MUTEX_UNLOCK(x)  pthread_mutex_unlock(&(x))
#define THREAD_ID        pthread_self()
 
/* This array will store all of the mutexes available to OpenSSL. */ 
static MUTEX_TYPE *mutex_buf = NULL;
 
static void locking_function(int mode, int n, const char *file, int line)
{
  if(mode & CRYPTO_LOCK)
    MUTEX_LOCK(mutex_buf[n]);
  else
    MUTEX_UNLOCK(mutex_buf[n]);
}
 
static unsigned long id_function(void)
{
	return ((unsigned long)syscall(__NR_gettid));
}
 
int _goboringcrypto_OPENSSL_thread_setup(void)
{
  int i;
 
  mutex_buf = malloc(_goboringcrypto_CRYPTO_num_locks() * sizeof(MUTEX_TYPE));
  if(!mutex_buf)
    return 0;
  for(i = 0;  i < _goboringcrypto_CRYPTO_num_locks();  i++)
    MUTEX_SETUP(mutex_buf[i]);
  _goboringcrypto_CRYPTO_set_id_callback(id_function);
  _goboringcrypto_CRYPTO_set_locking_callback(locking_function);
  return 1;
}
