// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that setgid does not hang on Linux.
// See https://golang.org/issue/3871 for details.

package cgotest

/*
#include <sys/types.h>
#include <unistd.h>
*/
import "C"

import (
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"
)

func runTestSetgid() bool {
	c := make(chan bool)
	go func() {
		C.setgid(0)
		c <- true
	}()
	select {
	case <-c:
		return true
	case <-time.After(5 * time.Second):
		return false
	}

}

func testSetgid(t *testing.T) {
	if !runTestSetgid() {
		t.Error("setgid hung")
	}

	// Now try it again after using signal.Notify.
	signal.Notify(make(chan os.Signal, 1), syscall.SIGINT)
	if !runTestSetgid() {
		t.Error("setgid hung after signal.Notify")
	}
}
