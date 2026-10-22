// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && goexperiment.arm64e

package runtime

// cfuncptr returns pc, the entry of a function that C calls, as a C
// function pointer. On arm64e that is pc signed with key IA and a zero
// modifier, because C calls function pointers through an
// authenticating branch.
//
// Implemented in sys_darwin_arm64.s.
func cfuncptr(pc uintptr) uintptr

// cfuncaddr is the inverse of cfuncptr: it authenticates the C function
// pointer p and returns the address that p points to.
//
// Implemented in sys_darwin_arm64.s.
func cfuncaddr(p uintptr) uintptr
