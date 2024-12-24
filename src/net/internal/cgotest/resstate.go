// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !netgo && cgo && darwin

package cgotest

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
