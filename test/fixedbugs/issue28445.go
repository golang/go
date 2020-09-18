// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var fp = (**float64)(nil)

func f() {
	switch fp {
	case new(*float64):
		println()
	}
}
