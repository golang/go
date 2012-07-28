// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that setgid does not hang on GNU/Linux.
// See http://code.google.com/p/go/issues/detail?id=3871 for details.

package cgotest

/*
#include <sys/types.h>
#include <unistd.h>
*/
import "C"

import (
	"testing"
	"time"
)

func testSetgid(t *testing.T) {
	c := make(chan bool)
	go func() {
		C.setgid(0)
		c <- true
	}()
	select {
	case <-c:
	case <-time.After(5 * time.Second):
		t.Error("setgid hung")
	}
}
