// true

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// a package that uses unsafe on the inside but not in it's api

package pkg

import "unsafe"

// this should be inlinable
func Float32bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}