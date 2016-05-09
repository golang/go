// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 2672
// was trying binary search with an interface type

package bug393

func f(x interface{}) int {
	switch x {
	case 1:
		return 1
	case 2:
		return 2
	case 3:
		return 3
	case 4:
		return 4
	case "5":
		return 5
	case "6":
		return 6
	default:
		return 7
	}
	panic("switch")
}
