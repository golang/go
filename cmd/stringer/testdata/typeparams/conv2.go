// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a version of ../conv.go with type params

// Check that constants defined as a conversion are accepted.

package main

import "fmt"

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type Other[T interface{ ~int | ~uint }] T // Imagine this is in another package.
type Other int

const (
	// alpha Other[int] = iota
	alpha Other = iota
	beta
	gamma
	delta
)

// type Conv2 Other[int]
type Conv2 Other

const (
	Alpha = Conv2(alpha)
	Beta  = Conv2(beta)
	Gamma = Conv2(gamma)
	Delta = Conv2(delta)
)

func main() {
	ck(Alpha, "Alpha")
	ck(Beta, "Beta")
	ck(Gamma, "Gamma")
	ck(Delta, "Delta")
	ck(42, "Conv2(42)")
}

func ck(c Conv2, str string) {
	if fmt.Sprint(c) != str {
		panic("conv2.go: " + str)
	}
}
