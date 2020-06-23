// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unsafeptr

import "unsafe"

func _() {
	var x unsafe.Pointer
	var y uintptr
	x = unsafe.Pointer(y) // ERROR "possible misuse of unsafe.Pointer"
	_ = x
}
