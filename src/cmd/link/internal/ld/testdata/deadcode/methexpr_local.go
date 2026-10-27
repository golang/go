// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// reflect.Value.MethodByName used as a function value in a local variable and
// called with a name the linker cannot see must keep all exported methods.

package main

import (
	"os"
	"reflect"
)

type T int

func (T) M() {}

func main() {
	f := reflect.Value.MethodByName
	f(reflect.ValueOf(T(1)), os.Args[0]).Call(nil)
}
