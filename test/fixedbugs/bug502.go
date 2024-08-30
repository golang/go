// build

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linking this with gccgo got an undefined symbol reference,
// because the private method in testing.TB led gccgo to assume that
// the interface method table would be defined in the testing package.

package main

import "testing"

type I interface {
	testing.TB
	Parallel()
}

func F(i I) {
	i.Log("F")
}

var t testing.T

func main() {
	F(&t)
}
