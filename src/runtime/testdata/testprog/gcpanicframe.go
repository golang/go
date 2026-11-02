// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

func init() {
	register("GCPanicFramePointer", GCPanicFramePointer)
}

type panicFrameBig struct {
	p   *[4]int
	pad [32]int // keeps the argument off the register path
}

var (
	panicFrameSink  *[4]int
	panicFrameValue int
)

//go:noinline
func panicFrameUse(p *[4]int) { panicFrameValue = p[0] }

//go:noinline
func panicFrameCall(fn func()) { fn() }

// panicFrameF takes its argument on the stack and lets a closure take its
// address, which makes the argument a stack object. The argument is dead once
// the closure has run, so the first collection frees what it points at. Goexit
// then records this frame in the panic, and the collection from the deferred
// call must not reach the argument through it.
//
//go:noinline
func panicFrameF(w panicFrameBig) {
	defer func() {
		runtime.GC()
		runtime.GC()
	}()
	panicFrameCall(func() { panicFrameUse(w.p) })
	panicFrameSink = nil
	runtime.GC()
	runtime.GC()
	runtime.Goexit()
}

func GCPanicFramePointer() {
	for i := 0; i < 200; i++ {
		done := make(chan struct{})
		go func() {
			defer close(done)
			p := new([4]int)
			p[0] = i
			panicFrameSink = p
			panicFrameF(panicFrameBig{p: p})
		}()
		<-done
	}
	fmt.Println("OK")
}
