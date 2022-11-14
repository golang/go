// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type T [129]byte

func main() {
	m := map[string]T{}
	v := reflect.ValueOf(m)
	v.SetMapIndex(reflect.ValueOf("a"), reflect.ValueOf(T{}))
	g = m["a"]
}

var g T
