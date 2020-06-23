// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test that the struct field in anonunion.go was promoted.
var v1 T
var v2 = v1.L

// Test that P, Q, and R all point to byte.
var v3 = Issue8478{P: (*byte)(nil), Q: (**byte)(nil), R: (***byte)(nil)}

// Test that N, A and B are fully defined
var v4 = N{}
var v5 = A{}
var v6 = B{}

// Test that S is fully defined
var v7 = S{}

// Test that #define'd type is fully defined
var _ = issue38649{X: 0}

func main() {
}
