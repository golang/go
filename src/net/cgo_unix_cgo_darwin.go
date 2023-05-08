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

func init() {
	const expected_size = unsafe.Sizeof(C.struct___res_state{})
	const got_size = unsafe.Sizeof(unix.ResState{})

	// This will cause a compile error when the size of
	// unix.ResState is too small.
	var _ [got_size - expected_size]byte
}
