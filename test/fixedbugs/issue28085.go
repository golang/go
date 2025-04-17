// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ = map[interface{}]int{
	0: 0,
	0: 0, // ERROR "duplicate"
}

var _ = map[interface{}]int{
	interface{}(0): 0,
	interface{}(0): 0, // ok
}

func _() {
	switch interface{}(0) {
	case 0:
	case 0: // ERROR "duplicate"
	}

	switch interface{}(0) {
	case interface{}(0):
	case interface{}(0): // ok
	}
}
