// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

// Test that a sequence of callbacks from C to Go get the same m.
// This failed to be true on arm and arm64, which was the root cause
// of issue 13881.

package main

/*
#include <stddef.h>
#include <pthread.h>

extern void GoCheckM();

static void* thread(void* arg __attribute__ ((unused))) {
	GoCheckM();
	return NULL;
}

static void CheckM() {
	pthread_t tid;
	pthread_create(&tid, NULL, thread, NULL);
	pthread_join(tid, NULL);
	pthread_create(&tid, NULL, thread, NULL);
	pthread_join(tid, NULL);
}
*/
import "C"

import (
	"fmt"
	"os"
)

func init() {
	register("EnsureDropM", EnsureDropM)
}

var savedM uintptr

//export GoCheckM
func GoCheckM() {
	m := runtime_getm_for_test()
	if savedM == 0 {
		savedM = m
	} else if savedM != m {
		fmt.Printf("m == %x want %x\n", m, savedM)
		os.Exit(1)
	}
}

func EnsureDropM() {
	C.CheckM()
	fmt.Println("OK")
}
