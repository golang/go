// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func spin() {
	var i int
	var b bool

	switch 1 {
	case 0:
		i = 1
	}
	switch 1 {
	case i:
	default:
		i = 1
		b = !b && (b && !b) && b
	}
	switch false {
	case false:
		i = 3 + -i
		switch 0 {
		case 1 - i:
		}
	}
}
