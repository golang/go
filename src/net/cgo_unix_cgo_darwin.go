// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !netgo && cgo && darwin

package net

/*
#include <resolv.h>
*/
import "C"

import (
	"internal/syscall/unix"
	"unsafe"
)

// This will cause a compile error when the size of
// unix.ResState is too small.
type _ [unsafe.Sizeof(unix.ResState{}) - unsafe.Sizeof(C.struct___res_state{})]byte

// This will cause a compile error when:
// unsafe.Sizeof(new(unix.ResState).Res_h_errno) != unsafe.Sizeof(new(C.struct___res_state).res_h_errno)
type _ [unsafe.Sizeof(new(unix.ResState).Res_h_errno) - unsafe.Sizeof(new(C.struct___res_state).res_h_errno)]byte
type _ [unsafe.Sizeof(new(C.struct___res_state).res_h_errno) - unsafe.Sizeof(new(unix.ResState).Res_h_errno)]byte

// This will cause a compile error when:
// unsafe.Offsetof(new(unix.ResState).Res_h_errno) != unsafe.Offsetof(new(C.struct___res_state).res_h_errno)
type _ [unsafe.Offsetof(new(unix.ResState).Res_h_errno) - unsafe.Offsetof(new(C.struct___res_state).res_h_errno)]byte
type _ [unsafe.Offsetof(new(C.struct___res_state).res_h_errno) - unsafe.Offsetof(new(unix.ResState).Res_h_errno)]byte
