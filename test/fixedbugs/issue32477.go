// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we use the deferreturn live map instead of
// the entry live map when handling a segv in a function
// that defers.

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

func gc(shouldFinalize bool) {
	runtime.GC()
	runtime.GC()
	runtime.GC()
	if shouldFinalize != finalized {
		err = "heap object finalized at the wrong time"
	}
}

func main() {
	h := new(HeapObj)
	h.init()
	runtime.SetFinalizer(h, func(h *HeapObj) {
		finalized = true
	})

	gc(false)
	g(h)
	if err != "" {
		panic(err)
	}
}

func g(h *HeapObj) {
	gc(false)
	h.check()
	// h is now unused
	defer func() {
		// h should not be live here. Previously we used to
		// use the function entry point as the place to get
		// the live map when handling a segv.
		gc(true)
		recover()
	}()
	*(*int)(nil) = 0 // trigger a segv
	return
}
