// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"strconv"
)

var trace string

func f() string {
	trace += "f"
	return "abc"
}

func g() *error {
	trace += "g"
	var x error
	return &x
}

func h() string {
	trace += "h"
	return "123"
}

func i() *int {
	trace += "i"
	var i int
	return &i
}

func main() {
	m := make(map[string]int)
	m[f()], *g() = strconv.Atoi(h())
	if m["abc"] != 123 || trace != "fgh" {
		println("BUG", m["abc"], trace)
		panic("fail")
	}
	mm := make(map[string]error)
	trace = ""
	mm["abc"] = os.EINVAL
	*i(), mm[f()] = strconv.Atoi(h())
	if mm["abc"] != nil || trace != "ifh" {
		println("BUG1", mm["abc"], trace)
		panic("fail")
	}
}
