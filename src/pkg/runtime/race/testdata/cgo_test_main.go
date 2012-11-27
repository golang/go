// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <pthread.h>

pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
int sync;

void Notify(void)
{
	pthread_mutex_lock(&mtx);
	sync = 1;
	pthread_cond_broadcast(&cv);
	pthread_mutex_unlock(&mtx);
}

void Wait(void)
{
	pthread_mutex_lock(&mtx);
	while(sync == 0)
		pthread_cond_wait(&cv, &mtx);
	pthread_mutex_unlock(&mtx);
}
*/
import "C"

func main() {
	data := 0
	go func() {
		data = 1
		C.Notify()
	}()
	C.Wait()
	_ = data
}
