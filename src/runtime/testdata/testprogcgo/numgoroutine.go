// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

package main

/*
#include <stddef.h>
#include <pthread.h>

extern void CallbackNumGoroutine();

static void* thread2(void* arg __attribute__ ((unused))) {
	CallbackNumGoroutine();
	return NULL;
}

static void CheckNumGoroutine() {
	pthread_t tid;
	pthread_create(&tid, NULL, thread2, NULL);
	pthread_join(tid, NULL);
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"strings"
)

var baseGoroutines int

func init() {
	register("NumGoroutine", NumGoroutine)
}

func NumGoroutine() {
	// Test that there are just the expected number of goroutines
	// running. Specifically, test that the spare M's goroutine
	// doesn't show up.
	if _, ok := checkNumGoroutine("first", 1+baseGoroutines); !ok {
		return
	}

	// Test that the goroutine for a callback from C appears.
	if C.CheckNumGoroutine(); !callbackok {
		return
	}

	// Make sure we're back to the initial goroutines.
	if _, ok := checkNumGoroutine("third", 1+baseGoroutines); !ok {
		return
	}

	fmt.Println("OK")
}

func checkNumGoroutine(label string, want int) (string, bool) {
	n := runtime.NumGoroutine()
	if n != want {
		fmt.Printf("%s NumGoroutine: want %d; got %d\n", label, want, n)
		return "", false
	}

	sbuf := make([]byte, 32<<10)
	sbuf = sbuf[:runtime.Stack(sbuf, true)]
	n = strings.Count(string(sbuf), "goroutine ")
	if n != want {
		fmt.Printf("%s Stack: want %d; got %d:\n%s\n", label, want, n, sbuf)
		return "", false
	}
	return string(sbuf), true
}

var callbackok bool

//export CallbackNumGoroutine
func CallbackNumGoroutine() {
	stk, ok := checkNumGoroutine("second", 2+baseGoroutines)
	if !ok {
		return
	}
	if !strings.Contains(stk, "CallbackNumGoroutine") {
		fmt.Printf("missing CallbackNumGoroutine from stack:\n%s\n", stk)
		return
	}

	callbackok = true
}
