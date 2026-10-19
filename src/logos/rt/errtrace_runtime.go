// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !logosejected

package rt

import "unsafe"

// Pulls of the runtime error-trace primitives. The definitions live in
// src/runtime/logos_errtrace.go and carry push-style //go:linkname marks,
// so the linker's linkname check accepts these references. Signatures must
// match the runtime side exactly.
//
// logosTraceRecord retains site in the goroutine's trace ring past return,
// so it is deliberately not //go:noescape: a retained site must escape to
// the heap, or a later stack move would leave a stale pointer behind.
// logosTraceRead only fills buf and retains nothing, so it is //go:noescape.

//go:linkname logosTraceRecord runtime.logosTraceRecord
func logosTraceRecord(site unsafe.Pointer, errData uintptr)

//go:linkname logosTraceRead runtime.logosTraceRead
//go:noescape
func logosTraceRead(buf []unsafe.Pointer) int

// errorWords mirrors the two words of a non-empty interface value: the
// type word followed by the data word. TraceErr reads only the data word
// as the error's identity, the same shape reflect uses for ifaceWords.
type errorWords struct {
	typ  unsafe.Pointer
	data uintptr
}

// TraceErr records s in the calling goroutine's error trace for err. A
// different error value resets the trace; the trace keeps the last 8
// sites in propagation order. It allocates nothing.
func TraceErr(err error, s *Site) {
	errData := (*errorWords)(unsafe.Pointer(&err)).data
	logosTraceRecord(unsafe.Pointer(s), errData)
}

// ErrorTrace returns the calling goroutine's recorded sites in
// propagation order, at most the last 8.
func ErrorTrace() []Site {
	var buf [8]unsafe.Pointer
	n := logosTraceRead(buf[:])
	out := make([]Site, n)
	for i := range out {
		out[i] = *(*Site)(buf[i])
	}
	return out
}
