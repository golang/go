// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5704: Conversions of empty strings to byte
// or rune slices return empty but non-nil slices.

package main

type (
	mystring string
	mybytes  []byte
	myrunes  []rune
)

func checkBytes(s []byte, arg string) {
	if len(s) != 0 {
		panic("len(" + arg + ") != 0")
	}
	if s == nil {
		panic(arg + " == nil")
	}
}

func checkRunes(s []rune, arg string) {
	if len(s) != 0 {
		panic("len(" + arg + ") != 0")
	}
	if s == nil {
		panic(arg + " == nil")
	}
}

func main() {
	checkBytes([]byte(""), `[]byte("")`)
	checkBytes([]byte(mystring("")), `[]byte(mystring(""))`)
	checkBytes(mybytes(""), `mybytes("")`)
	checkBytes(mybytes(mystring("")), `mybytes(mystring(""))`)

	checkRunes([]rune(""), `[]rune("")`)
	checkRunes([]rune(mystring("")), `[]rune(mystring(""))`)
	checkRunes(myrunes(""), `myrunes("")`)
	checkRunes(myrunes(mystring("")), `myrunes(mystring(""))`)
}
