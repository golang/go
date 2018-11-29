// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 21988: panic on switch case with invalid value

package p

const X = Wrong(0) // ERROR "undefined: Wrong"

func _() {
	switch 0 {
	case X:
	}
}
