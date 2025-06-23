// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Called from compiler-generated code; declared for go vet.
func udiv()
func _div()
func _divu()
func _mod()
func _modu()

// Called from assembly only; declared for go vet.
func usplitR0()
func load_g()
func save_g()
func emptyfunc()
func _initcgo()
func read_tls_fallback()

//go:noescape
func asmcgocall_no_g(fn, arg unsafe.Pointer)

// getfp returns the frame pointer register of its caller or 0 if not implemented.
// TODO: Make this a compiler intrinsic
func getfp() uintptr { return 0 }
