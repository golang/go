// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Use of public API is ok.

package main

import (
	"iter"
	"unique"
)

func seq(yield func(int) bool) {
	yield(123)
}

var s = "hello"

func main() {
	h := unique.Make(s)
	next, stop := iter.Pull(seq)
	defer stop()
	println(h.Value())
	println(next())
	println(next())
}
