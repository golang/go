// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Check that calls to C.malloc/C.free do not trigger TSAN false
// positive reports.

// #cgo CFLAGS: -fsanitize=thread
// #cgo LDFLAGS: -fsanitize=thread
// #include <stdlib.h>
import "C"

import (
	"runtime"
	"sync"
)

func main() {
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				p := C.malloc(C.size_t(i * 10))
				runtime.Gosched()
				C.free(p)
			}
		}()
	}
	wg.Wait()
}
