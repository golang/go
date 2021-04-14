// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Non-Go-constant but constant values aren't ok for array sizes.

package p

import "unsafe"

type T [uintptr(unsafe.Pointer(nil))]int // ERROR "non-constant array bound|array bound is not constant"

func f() {
	_ = complex(1<<uintptr(unsafe.Pointer(nil)), 0) // GCCGO_ERROR "non-integer type for left operand of shift"
}
