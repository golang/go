// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && !goexperiment.arm64e

package runtime

// cfuncptr returns pc, the entry of a function that C calls, as a C
// function pointer.
//
//go:nosplit
func cfuncptr(pc uintptr) uintptr { return pc }

// cfuncaddr returns the address that the C function pointer p points to.
//
//go:nosplit
func cfuncaddr(p uintptr) uintptr { return p }
