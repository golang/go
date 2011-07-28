// errchk $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ddd

func Sum() int
	for i := range []int{} { return i }  // ERROR "statement outside function|expected"

