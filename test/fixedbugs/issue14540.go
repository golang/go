// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(x int) {
	switch x {
	case 0:
		fallthrough
		; // ok
	case 1:
		fallthrough // ERROR "fallthrough statement out of place"
		{}
	case 2:
		fallthrough // ERROR "cannot fallthrough"
	}
}
