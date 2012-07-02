// compile

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3765

package main

func f(x uint) uint {
	m := ^(1 << x)
	return uint(m)
}
