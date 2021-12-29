// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

var V val[int]

type val[T any] struct {
	valx T
}

func (v *val[T]) Print() {
	v.print1()
}

func (v *val[T]) print1() {
	println(v.valx)
}

func (v *val[T]) fnprint1() {
	println(v.valx)
}

func FnPrint[T any](v *val[T]) {
	v.fnprint1()
}
