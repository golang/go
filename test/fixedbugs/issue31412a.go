// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code was incorrectly flagged as erroneous by gccgo.

package main

type Name string

type EFunc func(int) int

func Register(f EFunc, names ...Name) int {
	return f(len(names))
}

const (
	B Name = "B"
)

func RegisterIt() {
	n := B + "Duck"
	d := B + "Goose"
	f := func(x int) int { return x + 9 }
	Register(f, n, d)
}

func main() {
	RegisterIt()
}
