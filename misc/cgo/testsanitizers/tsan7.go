// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Setting an environment variable in a cgo program changes the C
// environment. Test that this does not confuse the race detector.

/*
#cgo CFLAGS: -fsanitize=thread
#cgo LDFLAGS: -fsanitize=thread
*/
import "C"

import (
	"fmt"
	"os"
	"sync"
	"time"
)

func main() {
	var wg sync.WaitGroup
	var mu sync.Mutex
	f := func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			time.Sleep(time.Microsecond)
			mu.Lock()
			s := fmt.Sprint(i)
			os.Setenv("TSAN_TEST"+s, s)
			mu.Unlock()
		}
	}
	wg.Add(2)
	go f()
	go f()
	wg.Wait()
}
