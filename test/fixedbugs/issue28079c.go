// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Non-Go-constant but constant values aren't ok for shifts.

package p

import "unsafe"

func f() {
	_ = complex(1<<uintptr(unsafe.Pointer(nil)), 0) // ERROR "invalid operation: shifted operand 1 \(type float64\) must be integer|non-integer type for left operand of shift"
}
