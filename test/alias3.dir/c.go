// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"./b"
)

func main() {
	var _ float64 = b.F(0)
	var _ a.Rune = int32(0)

	// embedded types can have different names but the same types
	var s a.S
	s.Int = 1
	s.IntAlias = s.Int
	s.IntAlias2 = s.Int

	// aliases denote identical types across packages
	var c a.Context = b.C
	var _ b.MyContext = c
}
