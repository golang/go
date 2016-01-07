// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

// Test that the Go runtime still works if C code changes the signal stack.

package cgotest

/*
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static stack_t oss;
static char signalStack[SIGSTKSZ];

static void changeSignalStack() {
	stack_t ss;
	memset(&ss, 0, sizeof ss);
	ss.ss_sp = signalStack;
	ss.ss_flags = 0;
	ss.ss_size = SIGSTKSZ;
	if (sigaltstack(&ss, &oss) < 0) {
		perror("sigaltstack");
		abort();
	}
}

static void restoreSignalStack() {
#if defined(__x86_64__) && defined(__APPLE__)
	// The Darwin C library enforces a minimum that the kernel does not.
	// This is OK since we allocated this much space in mpreinit,
	// it was just removed from the buffer by stackalloc.
	oss.ss_size = MINSIGSTKSZ;
#endif
	if (sigaltstack(&oss, NULL) < 0) {
		perror("sigaltstack restore");
		abort();
	}
}

static int zero() {
	return 0;
}
*/
import "C"

import (
	"runtime"
	"testing"
)

func testSigaltstack(t *testing.T) {
	switch {
	case runtime.GOOS == "solaris", runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64"):
		t.Skipf("switching signal stack not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	C.changeSignalStack()
	defer C.restoreSignalStack()
	defer func() {
		if recover() == nil {
			t.Error("did not see expected panic")
		}
	}()
	v := 1 / int(C.zero())
	t.Errorf("unexpected success of division by zero == %d", v)
}
