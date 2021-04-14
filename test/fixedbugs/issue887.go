// compile

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to leak registers on 8g.

package p

func f(x byte, y uint64) {
	var r byte
	switch y {
	case 1:
		r = x << y // '>>' triggers it too
	case 2:
		r = x << y
	case 3:
		r = x << y
	case 4:
		r = x << y
	case 5:
		r = x << y
	case 6:
		r = x << y
	case 7:
		r = x << y
	case 8:
		r = x << y
	case 9:
		r = x << y
	case 10:
		r = x << y
	}
	_ = r
}
