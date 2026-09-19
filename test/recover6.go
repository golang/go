// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that inlining a function that calls recover does not change
// recover semantics: gorecover counts logical (inline-expanded)
// frames between gopanic and gorecover, so an inlined recover must
// behave exactly like a non-inlined one.

package main

var got any

// doRecover and setGot are small enough to be inlined everywhere
// they are called.
func doRecover() any { return recover() }

func setGot() { got = recover() }

// calls recover two logical frames below its caller.
func setGotIndirect() { got = doRecover() }

// mustPanic runs f, which must panic with "boom".
func mustPanic(name string, f func()) {
	defer func() {
		if r := recover(); r != "boom" {
			panic(name + ": panic did not propagate")
		}
	}()
	f()
	panic(name + " did not panic")
}

// mustRecover runs f, which must recover from its own panic.
func mustRecover(name string, f func()) {
	defer func() {
		if r := recover(); r != nil {
			panic(name + ": panic escaped")
		}
	}()
	f()
}

func main() {
	// Directly deferred function calling recover: recovers.
	mustRecover("direct", func() {
		defer setGot()
		panic("boom")
	})
	if got != "boom" {
		panic("direct: recover did not see the panic value")
	}

	// doRecover inlined into the deferred closure: recover is
	// still two logical frames below gopanic and must return nil.
	mustPanic("closure", func() {
		defer func() { got = doRecover() }()
		panic("boom")
	})
	if got != nil {
		panic("closure: recover returned non-nil")
	}

	// Directly deferred function whose recover comes from an
	// inlined callee: recover is two logical frames below gopanic
	// and must return nil, exactly as if doRecover were not inlined.
	mustPanic("nested", func() {
		defer setGotIndirect()
		panic("boom")
	})
	if got != nil {
		panic("nested: recover returned non-nil")
	}

	// recover with no panic in flight, via an inlined helper.
	if r := doRecover(); r != nil {
		panic("idle: recover returned non-nil")
	}
}
