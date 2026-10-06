// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// reflect.Value.MethodByName as an element of a composite literal built in a
// function must keep all exported methods.

package main

import (
	"os"
	"reflect"
)

type T int

func (T) M() {}

func main() {
	table := []func(reflect.Value, string) reflect.Value{reflect.Value.MethodByName}
	table[0](reflect.ValueOf(T(1)), os.Args[0]).Call(nil)
}
