// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a method *is* live if it matches an interface
// method and the type is "indirectly" converted to an
// interface through reflection.

package main

import "reflect"

type I interface{ M() }

type T int

func (T) M() { println("XXX") }

func main() {
	e := reflect.ValueOf([]T{1}).Index(0).Interface()
	e.(I).M()
}
