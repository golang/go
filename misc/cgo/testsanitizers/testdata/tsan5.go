// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Check that calls to C.malloc/C.free do not collide with the calls
// made by the os/user package.

// #cgo CFLAGS: -fsanitize=thread
// #cgo LDFLAGS: -fsanitize=thread
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"os"
	"os/user"
	"runtime"
	"sync"
)

func main() {
	u, err := user.Current()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		// Let the test pass.
		os.Exit(0)
	}

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				user.Lookup(u.Username)
				runtime.Gosched()
			}
		}()
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				p := C.malloc(C.size_t(len(u.Username) + 1))
				runtime.Gosched()
				C.free(p)
			}
		}()
	}
	wg.Wait()
}
