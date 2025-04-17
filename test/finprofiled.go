// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that tiny allocations with finalizers are correctly profiled.
// Previously profile special records could have been processed prematurely
// (while the object is still live).

package main

import (
	"runtime"
	"time"
	"unsafe"
)

func main() {
	runtime.MemProfileRate = 1
	// Allocate 1M 4-byte objects and set a finalizer for every third object.
	// Assuming that tiny block size is 16, some objects get finalizers setup
	// only for middle bytes. The finalizer resurrects that object.
	// As the result, all allocated memory must stay alive.
	const (
		N             = 1 << 20
		tinyBlockSize = 16 // runtime._TinySize
	)
	hold := make([]*int32, 0, N)
	for i := 0; i < N; i++ {
		x := new(int32)
		if i%3 == 0 {
			runtime.SetFinalizer(x, func(p *int32) {
				hold = append(hold, p)
			})
		}
	}
	// Finalize as much as possible.
	// Note: the sleep only increases probability of bug detection,
	// it cannot lead to false failure.
	for i := 0; i < 5; i++ {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	// Read memory profile.
	var prof []runtime.MemProfileRecord
	for {
		if n, ok := runtime.MemProfile(prof, false); ok {
			prof = prof[:n]
			break
		} else {
			prof = make([]runtime.MemProfileRecord, n+10)
		}
	}
	// See how much memory in tiny objects is profiled.
	var totalBytes int64
	for _, p := range prof {
		bytes := p.AllocBytes - p.FreeBytes
		nobj := p.AllocObjects - p.FreeObjects
		if nobj == 0 {
			// There may be a record that has had all of its objects
			// freed. That's fine. Avoid a divide-by-zero and skip.
			continue
		}
		size := bytes / nobj
		if size == tinyBlockSize {
			totalBytes += bytes
		}
	}
	// 2*tinyBlockSize slack is for any boundary effects.
	if want := N*int64(unsafe.Sizeof(int32(0))) - 2*tinyBlockSize; totalBytes < want {
		println("got", totalBytes, "want >=", want)
		panic("some of the tiny objects are not profiled")
	}
	// Just to keep hold alive.
	if len(hold) != 0 && hold[0] == nil {
		panic("bad")
	}
}
