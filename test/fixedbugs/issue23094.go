// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that the array is reported in correct notation.

package p

var a [len(a)]int // ERROR "\[len\(a\)\]int"
