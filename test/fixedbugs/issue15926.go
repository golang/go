// build

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 15926: linker was adding .def to the end of symbols, causing
// a name collision with a method actually named def.

package main

type S struct{}

func (s S) def() {}

var I = S.def

func main() {
    I(S{})
}
