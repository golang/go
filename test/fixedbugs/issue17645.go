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
	var _ string = append(s, Foo{""}) // ERROR "cannot use append\(s, Foo{…}\) .* as string value in variable declaration" "cannot use Foo{…} .* as int value in argument to append" "cannot use .* as int value in struct literal"
}
