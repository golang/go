// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64

package main

// Test that an external C thread that is calling malloc can be hit
// with SIGCHLD signals. This used to fail when built with the race
// detector, because in that case the signal handler would indirectly
// call the C malloc function.

/*
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#define ALLOCERS 100
#define SIGNALERS 10

static void* signalThread(void* p) {
	pthread_t* pt = (pthread_t*)(p);
	int i, j;

	for (i = 0; i < 100; i++) {
		for (j = 0; j < ALLOCERS; j++) {
			if (pthread_kill(pt[j], SIGCHLD) < 0) {
				return NULL;
			}
		}
		usleep(1);
	}
	return NULL;
}

#define CALLS 100

static void* mallocThread(void* p) {
	int i;
	void *a[CALLS];

	for (i = 0; i < ALLOCERS; i++) {
		sched_yield();
	}
	for (i = 0; i < CALLS; i++) {
		a[i] = malloc(i);
	}
	for (i = 0; i < CALLS; i++) {
		free(a[i]);
	}
	return NULL;
}

void runRaceSignalThread() {
	int i;
	pthread_t m[ALLOCERS];
	pthread_t s[SIGNALERS];

	for (i = 0; i < ALLOCERS; i++) {
		pthread_create(&m[i], NULL, mallocThread, NULL);
	}
	for (i = 0; i < SIGNALERS; i++) {
		pthread_create(&s[i], NULL, signalThread, &m[0]);
	}
	for (i = 0; i < SIGNALERS; i++) {
		pthread_join(s[i], NULL);
	}
	for (i = 0; i < ALLOCERS; i++) {
		pthread_join(m[i], NULL);
	}
}
*/
import "C"

import (
	"fmt"
	"os"
	"time"
)

func init() {
	register("CgoRaceSignal", CgoRaceSignal)
}

func CgoRaceSignal() {
	// The failure symptom is that the program hangs because of a
	// deadlock in malloc, so set an alarm.
	go func() {
		time.Sleep(5 * time.Second)
		fmt.Println("Hung for 5 seconds")
		os.Exit(1)
	}()

	C.runRaceSignalThread()
	fmt.Println("OK")
}
