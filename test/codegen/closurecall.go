// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func closureCallCleanup() {}

func callKnownClosure(value, x int) int {
	fn := func(delta int) int {
		defer closureCallCleanup()
		return value + delta
	}
	// amd64:`CALL.*callKnownClosure.func1\(SB\)`
	// arm64:`CALL.*callKnownClosure.func1\(SB\)`
	return fn(x)
}

func callWithFunc(fn func(int) int, x int) int {
	return fn(x)
}

func callClosureAfterInlining(value, x int) int {
	fn := func(delta int) int {
		defer closureCallCleanup()
		return value + delta
	}
	// amd64:`CALL.*callClosureAfterInlining.func1\(SB\)`
	// arm64:`CALL.*callClosureAfterInlining.func1\(SB\)`
	return callWithFunc(fn, x)
}
