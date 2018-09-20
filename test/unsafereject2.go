// errorcheck -u -+

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that we cannot import the "unsafe" package when -u is supplied.

package a

import "unsafe" // ERROR "import package unsafe"

func Float32bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}
