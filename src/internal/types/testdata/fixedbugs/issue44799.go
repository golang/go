// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func Map[F, T any](s []F, f func(F) T) []T { return nil }

func Reduce[Elem1, Elem2 any](s []Elem1, initializer Elem2, f func(Elem2, Elem1) Elem2) Elem2 { var x Elem2; return x }

func main() {
	var s []int
	var f1 func(int) float64
	var f2 func(float64, int) float64
	_ = Map[int](s, f1)
	_ = Map(s, f1)
	_ = Reduce[int](s, 0, f2)
	_ = Reduce(s, 0, f2)
}
