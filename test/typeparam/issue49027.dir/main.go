// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"a"
	"fmt"
)

func main() {
	s := "foo"
	x := a.Conv(s)
	if x != s {
		panic(fmt.Sprintf("got %s wanted %s", x, s))
	}
	y, ok := a.Conv2(s)
	if !ok {
		panic("conversion failed")
	}
	if y != s {
		panic(fmt.Sprintf("got %s wanted %s", y, s))
	}
	z := a.Conv3(s)
	if z != s {
		panic(fmt.Sprintf("got %s wanted %s", z, s))
	}
	w := a.Conv4(a.Mystring(s))
	if w != a.Mystring(s) {
		panic(fmt.Sprintf("got %s wanted %s", w, s))
	}
}
