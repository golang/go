// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

package main

/*
#include <signal.h>
#include <stdlib.h>
#include <string.h>

static void abrthandler(int signum) {
	if (signum == SIGABRT) {
		exit(0);  // success
	}
}

void registerAbortHandler() {
	struct sigaction act;
	memset(&act, 0, sizeof act);
	act.sa_handler = abrthandler;
	sigaction(SIGABRT, &act, NULL);
}

static void __attribute__ ((constructor)) sigsetup(void) {
	if (getenv("CGOCATCHPANIC_EARLY_HANDLER") == NULL)
		return;
	registerAbortHandler();
}
*/
import "C"
import "os"

func init() {
	register("CgoCatchPanic", CgoCatchPanic)
}

// Test that the SIGABRT raised by panic can be caught by an early signal handler.
func CgoCatchPanic() {
	if _, ok := os.LookupEnv("CGOCATCHPANIC_EARLY_HANDLER"); !ok {
		C.registerAbortHandler()
	}
	panic("catch me")
}
