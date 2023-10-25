// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

import (
	"os"
	"runtime"
	"sync/atomic"
)

var started int32

// Start a goroutine that loops forever.
func init() {
	runtime.GOMAXPROCS(1)
	go func() {
		for {
			atomic.StoreInt32(&started, 1)
		}
	}()
}

//export GoFunction8
func GoFunction8() {
	for atomic.LoadInt32(&started) == 0 {
		runtime.Gosched()
	}
	os.Exit(0)
}

func main() {
}
