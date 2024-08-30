// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4614: slicing of nil slices confuses the compiler
// with a uintptr(nil) node.

package p

import "unsafe"

var n int

var _ = []int(nil)[1:]
var _ = []int(nil)[n:]

var _ = uintptr(unsafe.Pointer(nil))
var _ = unsafe.Pointer(uintptr(0))
