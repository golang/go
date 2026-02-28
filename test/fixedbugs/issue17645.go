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
	var _ string = append(s, Foo{""}) // ERROR "cannot use .. \(.*untyped string.*\) as .*int.*|incompatible type" "cannot use Foo{.*} \(.*type Foo\) as type int in .*append" "cannot use append\(s\, Foo{.*}\) \(.*type \[\]int\) as type string in (assignment|variable declaration)"
}
