// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type X struct {
	T [32]byte
}

func (x *X) Get() []byte {
	t := x.T
	return t[:]
}

func (x *X) RetPtr(i int) *int {
	i++
	return &i
}

func (x *X) RetRPtr(i int) (r1 int, r2 *int) {
	r1 = i + 1
	r2 = &r1
	return
}
