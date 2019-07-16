// compile

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x

type T struct {
	i int
	e interface{}
}

func (t *T) F() bool {
	if t.i != 0 {
		return false
	}
	_, ok := t.e.(string)
	return ok
}

var x int

func g(t *T) {
	if t.F() || true {
		if t.F() {
			x = 0
		}
	}
}
