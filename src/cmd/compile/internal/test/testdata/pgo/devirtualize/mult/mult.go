// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// WARNING: Please avoid updating this file.
// See the warning in ../devirt.go for more details.

package mult

var sink int

type Multiplier interface {
	Multiply(a, b int) int
}

type Mult struct{}

func (Mult) Multiply(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a * b
}

type NegMult struct{}

func (NegMult) Multiply(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return -1 * a * b
}
