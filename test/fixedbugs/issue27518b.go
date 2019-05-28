// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

var finalized bool
var err string

type HeapObj [8]int64

const filler int64 = 0x123456789abcdef0

func (h *HeapObj) init() {
	for i := 0; i < len(*h); i++ {
		h[i] = filler
	}
}
func (h *HeapObj) check() {
	for i := 0; i < len(*h); i++ {
		if h[i] != filler {
			err = "filler overwritten"
		}
	}
}

type StackObj struct {
	h *HeapObj
}

func gc(shouldFinalize bool) {
	runtime.GC()
	runtime.GC()
	runtime.GC()
	if shouldFinalize != finalized {
		err = "heap object finalized at the wrong time"
	}
}

func main() {
	var s StackObj
	s.h = new(HeapObj)
	s.h.init()
	runtime.SetFinalizer(s.h, func(h *HeapObj) {
		finalized = true
	})
	gc(false)
	h := g(&s)
	gc(false)
	h.check()
	gc(true) // finalize here, after return value's last use. (Go1.11 never runs the finalizer.)
	if err != "" {
		panic(err)
	}
}

func g(p *StackObj) (v *HeapObj) {
	gc(false)
	v = p.h // last use of the stack object. the only reference to the heap object is in the return slot.
	gc(false)
	defer func() {
		gc(false)
		recover()
		gc(false)
	}()
	*(*int)(nil) = 0
	return
}
