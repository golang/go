// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Check that writes to Go allocated memory, with Go synchronization,
// do not look like a race.

/*
#cgo CFLAGS: -fsanitize=thread
#cgo LDFLAGS: -fsanitize=thread

void f(char *p) {
	*p = 1;
}
*/
import "C"

import (
	"runtime"
	"sync"
)

func main() {
	var wg sync.WaitGroup
	var mu sync.Mutex
	c := make(chan []C.char, 100)
	for i := 0; i < 10; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				c <- make([]C.char, 4096)
				runtime.Gosched()
			}
		}()
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				p := &(<-c)[0]
				mu.Lock()
				C.f(p)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
}
