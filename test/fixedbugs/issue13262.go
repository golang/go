// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 13262: cmd/compile: bogus "fallthrough
// statement out of place" error

package p

func f() int {
	var a int
	switch a {
	case 0:
		return func() int { return 1 }()
		fallthrough
	default:
	}
	return 0
}
