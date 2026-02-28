// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !android
// +build !android

// Test that pthread_cancel works as expected
// (NPTL uses SIGRTMIN to implement thread cancellation)
// See https://golang.org/issue/6997
package cgotest

/*
#cgo CFLAGS: -pthread
#cgo LDFLAGS: -pthread
extern int StartThread();
extern int CancelThread();
*/
import "C"

import (
	"testing"
	"time"
)

func test6997(t *testing.T) {
	r := C.StartThread()
	if r != 0 {
		t.Error("pthread_create failed")
	}
	c := make(chan C.int)
	go func() {
		time.Sleep(500 * time.Millisecond)
		c <- C.CancelThread()
	}()

	select {
	case r = <-c:
		if r == 0 {
			t.Error("pthread finished but wasn't canceled??")
		}
	case <-time.After(30 * time.Second):
		t.Error("hung in pthread_cancel/pthread_join")
	}
}
