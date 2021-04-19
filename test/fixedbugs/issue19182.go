// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"time"
)

var a uint64 = 0

func main() {
	runtime.GOMAXPROCS(2) // With just 1, infinite loop never yields

	go func() {
		for {
			atomic.AddUint64(&a, uint64(1))
		}
	}()

	time.Sleep(10 * time.Millisecond) // Short sleep is enough in passing case
	i, val := 0, atomic.LoadUint64(&a)
	for ; val == 0 && i < 100; val, i = atomic.LoadUint64(&a), i+1 {
		time.Sleep(100 * time.Millisecond)
	}
	if val == 0 {
		fmt.Printf("Failed to observe atomic increment after %d tries\n", i)
	}

}
