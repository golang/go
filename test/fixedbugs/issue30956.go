// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for compile generated static data for literal
// composite struct

package main

import "fmt"

type X struct {
	V interface{}

	a int
	b int
	c int
}

func pr(x X) {
	fmt.Println(x.V)
}

func main() {
	pr(X{
		V: struct {
			A int
		}{42},
	})
}
