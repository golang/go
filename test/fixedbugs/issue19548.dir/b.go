// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

type Value interface {
	a.Stringer
	Addr() *a.Mode
}

var global a.Mode

func f() int {
	var v Value
	v = &global
	return int(v.String()[0])
}

func main() {
	f()
}
