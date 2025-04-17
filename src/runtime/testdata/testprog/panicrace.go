// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"sync"
)

func init() {
	register("PanicRace", PanicRace)
}

func PanicRace() {
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer func() {
			wg.Done()
			runtime.Gosched()
		}()
		panic("crash")
	}()
	wg.Wait()
}
