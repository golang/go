// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles in the method value of an embedded struct returned
// from a function call.

package funcembedmethvalue

type T int

func (T) m() int {
	_ = x
	return 0
}

func g() E {
	return E{0}
}

type E struct{ T }

var (
	e E
	x = g().m // ERROR "initialization cycle|depends upon itself" 
)
