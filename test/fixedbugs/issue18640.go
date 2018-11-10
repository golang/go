// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
	a = b
	b struct {
		*a
	}

	c struct {
		*d
	}
	d = c

	e = f
	f = g
	g = []h
	h i
	i = j
	j = e
)
