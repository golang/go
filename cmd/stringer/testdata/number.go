// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Enumeration with an offset.
// Also includes a duplicate.

package main

import "fmt"

type Number int

const (
	_ Number = iota
	One
	Two
	Three
	AnotherOne = One // Duplicate; note that AnotherOne doesn't appear below.
)

func main() {
	ck(One, "One")
	ck(Two, "Two")
	ck(Three, "Three")
	ck(AnotherOne, "One")
	ck(127, "Number(127)")
}

func ck(num Number, str string) {
	if fmt.Sprint(num) != str {
		panic("number.go: " + str)
	}
}
