// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
)

var x = struct{ a, _, c int }{1, 2, 3}

func main() {
	if i := reflect.ValueOf(x).Field(1).Int(); i != 0 {
		println("got", i, "want", 0)
		panic("fail")
	}
}
