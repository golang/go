// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f_ssa(x int, p *int) {
	if false {
		y := x + 5
		for {
			*p = y
		}
	}
}
