// cmpout

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that println can be the target of a go statement.

package main

import (
	"runtime"
	"time"
)

func main() {
	go println(42, true, false, true, 1.5, "world", (chan int)(nil), []int(nil), (map[string]int)(nil), (func())(nil), byte(255))
	for runtime.NumGoroutine() > 1 {
		time.Sleep(10*time.Millisecond)
	}
}
