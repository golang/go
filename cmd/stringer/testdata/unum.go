// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unsigned integers spanning zero.

package main

import "fmt"

type Unum uint8

const (
	m_2 Unum = iota + 253
	m_1
)

const (
	m0 Unum = iota
	m1
	m2
)

func main() {
	ck(^Unum(0)-3, "Unum(252)")
	ck(m_2, "m_2")
	ck(m_1, "m_1")
	ck(m0, "m0")
	ck(m1, "m1")
	ck(m2, "m2")
	ck(3, "Unum(3)")
}

func ck(unum Unum, str string) {
	if fmt.Sprint(unum) != str {
		panic("unum.go: " + str)
	}
}
