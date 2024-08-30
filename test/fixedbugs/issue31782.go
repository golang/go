// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check static composite literal reports wrong for struct
// field.

package main

type one struct {
	i interface{}
}

type two struct {
	i interface{}
	s []string
}

func main() {
	o := one{i: two{i: 42}.i}
	println(o.i.(int))
}
