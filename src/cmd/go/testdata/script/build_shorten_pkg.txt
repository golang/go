[short] skip

# This test may go away when the loopvar experiment goes away.
# Accurate reporting of notable loops in the presence of inlining
# can create warnings in sibling directories, and it's nice if those
# can be trimmed like subdirectory paths are.

env GOEXPERIMENT=loopvar
go build -gcflags=inlines/a=-d=loopvar=2 .
stderr ^\.[\\/]b[\\/]b\.go:12:6:.*loop.inlined.into.a[\\/]a\.go
stderr ^\.[\\/]b[\\/]b\.go:12:9:.*loop.inlined.into.a[\\/]a\.go

-- go.mod --
module inlines

go 1.21
-- a/a.go --
// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "inlines/b"

func F() []*int {
	var s []*int
	for i := 0; i < 10; i++ {
		s = append(s, &i)
	}
	return s
}

func Fb() []*int {
	bf, _ := b.F()
	return bf
}
-- b/b.go --
package b

var slice = []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}

func F() ([]*int, []*int) {
	return g()
}

func g() ([]*int, []*int) {
	var s []*int
	var t []*int
	for i, j := range slice {
		s = append(s, &i)
		t = append(t, &j)
	}
	return s[:len(s)-1], t
}
-- main.go --
package main

import (
	"fmt"
	"inlines/a"
	"inlines/b"
)

func sum(s []*int) int {
	sum := 0
	for _, pi := range s {
		sum += *pi
	}
	return sum
}

func main() {
	af := a.F()
	bf, _ := b.F()
	abf := a.Fb()

	saf, sbf, sabf := sum(af), sum(bf), sum(abf)

	fmt.Printf("af, bf, abf sums = %d, %d, %d\n", saf, sbf, sabf)
}
