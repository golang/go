// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2627 -- unsafe.Pointer type isn't handled nicely in some errors

package main

import "unsafe"

func main() {
	var x *int
	_ = unsafe.Pointer(x) - unsafe.Pointer(x) // ERROR "operator - not defined on unsafe.Pointer|expected integer, floating, or complex type"
}
