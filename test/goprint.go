// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that println can be the target of a go statement.

package main

import (
	"log"
	"runtime"
	"time"
)

func main() {
	numg0 := runtime.NumGoroutine()
	deadline := time.Now().Add(10 * time.Second)
	go println(42, true, false, true, 1.5, "world", (chan int)(nil), []int(nil), (map[string]int)(nil), (func())(nil), byte(255))
	for {
		numg := runtime.NumGoroutine()
		if numg > numg0 {
			if time.Now().After(deadline) {
				log.Fatalf("%d goroutines > initial %d after deadline", numg, numg0)
			}
			runtime.Gosched()
			continue
		}
		break
	}
}
