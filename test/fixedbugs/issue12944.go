// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

const (
	_ = unsafe.Sizeof([0]byte{}[0]) // ERROR "out of bounds"
)
