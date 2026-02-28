// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles in a method call.

package methcall

type T int

func (T) m() int {
	_ = x
	return 0
}

var (
	t T
	x = t.m() // ERROR "initialization cycle|depends upon itself"
)
