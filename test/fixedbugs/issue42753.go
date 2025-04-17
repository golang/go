// compile -d=ssa/check/on

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() uint32 {
	s := "\x01"
	x := -int32(s[0])
	return uint32(x) & 0x7fffffff
}
