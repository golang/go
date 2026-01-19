// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimesecret && (arm64 || amd64) && linux

package secret_test

import (
	"runtime"
	"runtime/secret"
	"testing"
)

func TestInterleavedAllocFrees(t *testing.T) {
	// Interleave heap objects that are kept alive beyond secret.Do
	// with heap objects that do not live past secret.Do.
	// The intent is for the clearing of one object (with the wrong size)
	// to clobber the type header of the next slot. If the GC sees a nil type header
	// when it expects to find one, it can throw.
	type T struct {
		p *int
		x [1024]byte
	}
	for range 10 {
		var s []*T
		secret.Do(func() {
			for i := range 100 {
				t := &T{}
				if i%2 == 0 {
					s = append(s, t)
				}
			}
		})
		runtime.GC()
		runtime.GC()
		runtime.KeepAlive(s)
	}
}
