// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gccgo gave an invalid error ("floating point constant truncated to
// integer") compiling this.

package p

const C = 1<<63 - 1

func F(i int64) int64 {
	return i
}

var V = F(int64(C) / 1e6)
