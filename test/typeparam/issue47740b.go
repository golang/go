// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type S[T any] struct {
	a interface{}
}

func (e S[T]) M() {
	v := reflect.ValueOf(e.a)
	_, _ = v.Interface().(int)
}

func main() {
	e := S[int]{0}
	e.M()
}
