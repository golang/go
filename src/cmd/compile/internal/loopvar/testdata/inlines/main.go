// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/compile/internal/loopvar/testdata/inlines/a"
	"cmd/compile/internal/loopvar/testdata/inlines/b"
	"cmd/compile/internal/loopvar/testdata/inlines/c"
	"fmt"
	"os"
)

func sum(s []*int) int {
	sum := 0
	for _, pi := range s {
		sum += *pi
	}
	return sum
}

var t []*int

func F() []*int {
	var s []*int
	for i, j := 0, 0; j < 10; i, j = i+1, j+1 {
		s = append(s, &i)
		t = append(s, &j)
	}
	return s
}

func main() {
	f := F()
	af := a.F()
	bf, _ := b.F()
	abf := a.Fb()
	cf := c.F()

	sf, saf, sbf, sabf, scf := sum(f), sum(af), sum(bf), sum(abf), sum(cf)

	fmt.Printf("f, af, bf, abf, cf sums = %d, %d, %d, %d, %d\n", sf, saf, sbf, sabf, scf)

	// Special failure just for use with hash searching, to prove it fires exactly once.
	// To test: `gossahash -e loopvarhash go run .` in this directory.
	// This is designed to fail in two different ways, because gossahash searches randomly
	// it will find both failures over time.
	if os.Getenv("GOCOMPILEDEBUG") != "" && (sabf == 45 || sf == 45) {
		os.Exit(11)
	}
	os.Exit(0)
}
