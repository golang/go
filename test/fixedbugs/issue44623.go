// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// An open-coded defer must not keep its closure reachable after the
// deferred call has returned.

package main

import "runtime"

var gone = make(chan bool, 1)

// check runs last, after use(p) has returned.
func check() {
	runtime.GC()
	runtime.GC()
	// Deadlocks if the finalizer did not run.
	<-gone
}

//go:noinline
func use(p *[64]byte) {}

func main() {
	// Does not work with gccgo, which has a partially conservative GC.
	if runtime.Compiler == "gccgo" {
		return
	}
	p := new([64]byte)
	runtime.SetFinalizer(p, func(*[64]byte) { gone <- true })
	defer check()
	defer use(p)
}
