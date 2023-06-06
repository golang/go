// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

func main() {
	type MyByte byte
	type MyRune rune
	type MyString string

	a := []MyByte{'a', 'b', 'c'}
	if s := string(a); s != "abc" {
		panic(s)
	}

	b := []MyRune{'五', '五'}
	if s := string(b); s != "五五" {
		panic(s)
	}

	c := []MyByte{'l', 'o', 'r', 'e', 'm'}
	if s := MyString(c); s != MyString("lorem") {
		panic(s)
	}

	d := "lorem"
	if a := []MyByte(d); !reflect.DeepEqual(a, []MyByte{'l', 'o', 'r', 'e', 'm'}) {
		panic(a)
	}

	e := 42
	if s := MyString(e); s != "*" {
		panic(s)
	}
}
