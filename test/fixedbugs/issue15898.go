// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(e interface{}) {
	switch e.(type) {
	case nil, nil: // ERROR "multiple nil cases in type switch|duplicate type in switch"
	}

	switch e.(type) {
	case nil:
	case nil: // ERROR "multiple nil cases in type switch|duplicate type in switch"
	}
}
