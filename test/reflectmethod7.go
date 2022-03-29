// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// See issue 44207.

package main

import "reflect"

type S int

func (s S) M() {}

func main() {
	t := reflect.TypeOf(S(0))
	fn, ok := reflect.PointerTo(t).MethodByName("M")
	if !ok {
		panic("FAIL")
	}
	fn.Func.Call([]reflect.Value{reflect.New(t)})
}
