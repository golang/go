// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles when calling an embedded method expression.

package embedmethexprcall

type T int

func (T) m() int {
	_ = x
	return 0
}

type E struct{ T }

var x = E.m(E{0}) // ERROR "initialization loop|depends upon itself" 
