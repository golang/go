// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// reflect.Value.Method used as a function value must keep all exported
// methods, like a direct call with a non-constant index does.

package main

import (
	"os"
	"reflect"
)

type T int

func (T) M() {}

func main() {
	f := reflect.Value.Method
	f(reflect.ValueOf(T(1)), len(os.Args)-1).Call(nil)
}
