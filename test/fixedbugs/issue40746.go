// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(x byte, b bool) byte {
	var c byte
	if b {
		c = 1
	}

	if int8(c) < 0 {
		x++
	}
	return x
}
