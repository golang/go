// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package main

import (
	"fmt"
	"os"
)

/*
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

sig_atomic_t expectCSigsegv;
int *sigfwdP;

static void sigsegv() {
	expectCSigsegv = 1;
	*sigfwdP = 1;
	fprintf(stderr, "ERROR: C SIGSEGV not thrown on caught?.\n");
	exit(2);
}

static void segvhandler(int signum) {
	if (signum == SIGSEGV) {
		if (expectCSigsegv == 0) {
			fprintf(stderr, "SIGSEGV caught in C unexpectedly\n");
			exit(1);
		}
		fprintf(stdout, "OK\n");
		exit(0);  // success
	}
}

static void __attribute__ ((constructor)) sigsetup(void) {
	if (getenv("GO_TEST_CGOSIGFWD") == NULL) {
		return;
	}

	struct sigaction act;

	memset(&act, 0, sizeof act);
	act.sa_handler = segvhandler;
	sigaction(SIGSEGV, &act, NULL);
}
*/
import "C"

func init() {
	register("CgoSigfwd", CgoSigfwd)
}

var nilPtr *byte

func f() (ret bool) {
	defer func() {
		if recover() == nil {
			fmt.Fprintf(os.Stderr, "ERROR: couldn't raise SIGSEGV in Go\n")
			C.exit(2)
		}
		ret = true
	}()
	*nilPtr = 1
	return false
}

func CgoSigfwd() {
	if os.Getenv("GO_TEST_CGOSIGFWD") == "" {
		fmt.Fprintf(os.Stderr, "test must be run with GO_TEST_CGOSIGFWD set\n")
		os.Exit(1)
	}

	// Test that the signal originating in Go is handled (and recovered) by Go.
	if !f() {
		fmt.Fprintf(os.Stderr, "couldn't recover from SIGSEGV in Go.\n")
		C.exit(2)
	}

	// Test that the signal originating in C is handled by C.
	C.sigsegv()
}
