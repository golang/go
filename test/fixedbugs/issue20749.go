// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Verify that the compiler complains even if the array
// has length 0.
var a [0]int
var _ = a[2:] // ERROR "invalid slice index 2"

var b [1]int
var _ = b[2:] // ERROR "invalid slice index 2"
