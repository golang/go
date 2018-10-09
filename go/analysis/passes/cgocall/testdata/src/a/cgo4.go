// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the cgo checker on a file that doesn't use cgo, but has an
// import named "C".

package a

import C "fmt"

var _ = C.Println(*p(**p))

// Passing a pointer (via a slice), but C is fmt, not cgo.
var _ = C.Println([]int{3})
