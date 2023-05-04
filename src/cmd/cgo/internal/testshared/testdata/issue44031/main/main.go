// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testshared/issue44031/b"

type t int

func (t) m() {}

type i interface{ m() } // test that unexported method is correctly marked

var v interface{} = t(0)

func main() {
	b.F()
	v.(i).m()
}
