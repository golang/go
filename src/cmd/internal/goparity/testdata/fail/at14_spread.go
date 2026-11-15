// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	A int
	B int
}

func main() {
	s := S{1, 2}
	t := S{..s}
	println(t.A)
}
