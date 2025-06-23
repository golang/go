// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles in a method expression call.

package methexprcall

type T int

func (T) m() int {
	_ = x
	return 0
}

var x = T.m(0) // ERROR "initialization cycle|depends upon itself"
