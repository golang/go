// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// panicOnCall is invoked by the compiler-inserted panic-on-call instrumentation.
// It panics unconditionally so that fuzzers observe the crash at the call site.
//
//go:noinline
func panicOnCall(p *byte, length uintptr) {
	s := unsafe.String(p, int(length))
	panic(errorString("panic-on-call: " + s))
}
