// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sort"
	"strings"

	"issue20014.dir/a"
)

func main() {
	samePackage()
	crossPackage()

	// Print fields registered with field tracking.
	var fields []string
	for _, line := range strings.Split(fieldTrackInfo, "\n") {
		if line != "" {
			fields = append(fields, strings.Split(line, "\t")[0])
		}
	}
	sort.Strings(fields) // for stable output, regardless of optimizations
	for _, field := range fields {
		println(field)
	}
}

type T struct {
	X int `go:"track"`
	Y int `go:"track"`
	Z int // untracked
}

func (t *T) GetX() int {
	return t.X
}
func (t *T) GetY() int {
	return t.Y
}
func (t *T) GetZ() int {
	return t.Z
}

func samePackage() {
	var t T
	println(t.GetX())
	println(t.GetZ())
}

func crossPackage() {
	var t a.T
	println(t.GetX())
	println(t.GetZ())
}

// This global variable is set by the linker using the -k option.
var fieldTrackInfo string
