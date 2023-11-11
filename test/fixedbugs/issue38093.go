// run

//go:build js

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test race condition between timers and wasm calls that led to memory corruption.

package main

import (
	"os"
	"syscall/js"
	"time"
)

func main() {
	ch1 := make(chan struct{})

	go func() {
		for {
			time.Sleep(5 * time.Millisecond)
			ch1 <- struct{}{}
		}
	}()
	go func() {
		for {
			time.Sleep(8 * time.Millisecond)
			ch1 <- struct{}{}
		}
	}()
	go func() {
		time.Sleep(2 * time.Second)
		os.Exit(0)
	}()

	for range ch1 {
		ch2 := make(chan struct{}, 1)
		f := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			ch2 <- struct{}{}
			return nil
		})
		defer f.Release()
		fn := js.Global().Get("Function").New("cb", "cb();")
		fn.Invoke(f)
		<-ch2
	}
}
