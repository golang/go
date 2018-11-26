// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Non-Go-constant but constant indexes are ok at compile time.

package p

import "unsafe"

func f() {
	var x [0]int
	x[uintptr(unsafe.Pointer(nil))] = 0
}
func g() {
	var x [10]int
	_ = x[3:uintptr(unsafe.Pointer(nil))]
}
