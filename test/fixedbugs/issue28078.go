// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Non-constant duplicate keys/cases should not be reported
// as errors by the compiler.

package p

import "unsafe"

func f() {
	_ = map[uintptr]int{
		0:                            0,
		uintptr(unsafe.Pointer(nil)): 0,
	}

	switch uintptr(0) {
	case 0:
	case uintptr(unsafe.Pointer(nil)):
	}

	switch interface{}(nil) {
	case nil:
	case nil:
	}

	_ = map[interface{}]int{
		nil: 0,
		nil: 0,
	}
}
