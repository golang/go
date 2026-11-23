// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !unix && !windows

package cgo

import _ "unsafe" // for go:linkname

// x_crosscall2_ptr and _crosscall2_ptr are referenced by asm_*.s
// (set_crosscall2) on targets without a Unix or Windows runtime/cgo
// implementation (such as Plan 9).
var x_crosscall2_ptr uintptr

//go:linkname _crosscall2_ptr _crosscall2_ptr
var _crosscall2_ptr = &x_crosscall2_ptr
