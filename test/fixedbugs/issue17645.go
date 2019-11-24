// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Foo struct {
	X int
}

func main() {
	var s []int
	var _ string = append(s, Foo{""}) // ERROR "cannot use .. \(type untyped string\) as type int in field value" "cannot use Foo literal \(type Foo\) as type int in append" "cannot use append\(s\, Foo literal\) \(type \[\]int\) as type string in assignment"
}
