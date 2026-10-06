// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// reflect.Value.MethodByName stored in a package variable by the static
// initializer must keep all exported methods.

package main

import (
	"os"
	"reflect"
)

type T int

func (T) M() {}

var f = reflect.Value.MethodByName

//go:noinline
func call(v reflect.Value, name string) { f(v, name).Call(nil) }

func main() { call(reflect.ValueOf(T(1)), os.Args[0]) }
