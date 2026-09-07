// run -gcflags=all=-d=checkptr=1

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type T struct{}

func (T) M() int { return 42 }

func main() {
	v := reflect.ValueOf(T{})
	// Both paths construct a function value from a method's text offset.
	// On wasm32, its code word is an integer function handle, not a pointer.
	if got := v.Type().Method(0).Func.Call([]reflect.Value{v})[0].Int(); got != 42 {
		panic("incorrect Type.Method result")
	}
	if got := v.Method(0).Call(nil)[0].Int(); got != 42 {
		panic("incorrect Value.Method result")
	}
}
