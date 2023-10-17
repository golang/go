// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"./b"
)

func main() {
	switch b.I.(type) {
	case a.G[b.T]:
	case int:
		panic("bad")
	case float64:
		panic("bad")
	default:
		panic("bad")
	}

	b.F(a.G[b.T]{})
}
