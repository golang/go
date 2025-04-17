// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

var (
	x uint
	y = x
	z = uintptr(y)
	a = uint32(y)
	b = uint64(y)
)
