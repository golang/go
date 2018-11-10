// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p1

type Magic int

type T struct {
	x interface{}
}

func (t *T) M() bool {
	_, ok := t.x.(Magic)
	return ok
}

func F(t *T) {
	println(t)
}
