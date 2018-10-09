// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the cgo checker on a file that doesn't use cgo.

package a

import "unsafe"

// Passing a pointer (via the slice), but C isn't cgo.
var _ = C.f(unsafe.Pointer(new([]int)))

var C struct{ f func(interface{}) int }
