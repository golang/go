// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	joao = "João"
	jose = "José"
)

func main() {
	s1 := joao
	s2 := jose
	if (s1 < s2) != (joao < jose) {
		panic("unequal")
	}
}
