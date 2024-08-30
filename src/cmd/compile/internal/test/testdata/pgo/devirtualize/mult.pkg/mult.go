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

// N.B. Different types than AddFunc to test intra-line disambiguation.
type MultFunc func(int64, int64) int64

func MultFn(a, b int64) int64 {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a * b
}

func NegMultFn(a, b int64) int64 {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return -1 * a * b
}

//go:noinline
func MultClosure() MultFunc {
	// Explicit closure to differentiate from AddClosure.
	c := 1
	return func(a, b int64) int64 {
		for i := 0; i < 1000; i++ {
			sink++
		}
		return a * b * int64(c)
	}
}

//go:noinline
func NegMultClosure() MultFunc {
	c := 1
	return func(a, b int64) int64 {
		for i := 0; i < 1000; i++ {
			sink++
		}
		return -1 * a * b * int64(c)
	}
}
