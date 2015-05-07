// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

/*
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

int *p;
static void sigsegv() {
	*p = 1;
	fprintf(stderr, "ERROR: C SIGSEGV not thrown on caught?.\n");
	exit(2);
}

static void sighandler(int signum) {
	if (signum == SIGSEGV) {
		exit(0);  // success
	}
}

static void __attribute__ ((constructor)) sigsetup(void) {
	struct sigaction act;
	act.sa_handler = &sighandler;
	sigaction(SIGSEGV, &act, 0);
}
*/
import "C"

var p *byte

func f() (ret bool) {
	defer func() {
		if recover() == nil {
			fmt.Errorf("ERROR: couldn't raise SIGSEGV in Go.")
			C.exit(2)
		}
		ret = true
	}()
	*p = 1
	return false
}

func main() {
	// Test that the signal originating in Go is handled (and recovered) by Go.
	if !f() {
		fmt.Errorf("couldn't recover from SIGSEGV in Go.")
		C.exit(2)
	}

	// Test that the signal originating in C is handled by C.
	C.sigsegv()
}
