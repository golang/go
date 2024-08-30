// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 22063: panic on interface switch case with invalid name

package p

const X = Wrong(0) // ERROR "undefined: Wrong|reference to undefined name .*Wrong"

func _() {
	switch interface{}(nil) {
	case X:
	}
}
