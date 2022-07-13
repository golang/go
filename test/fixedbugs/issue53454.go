// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T1 struct {
	A T5
	B T2
	C T7
	D T4
}

type T2 struct {
	T3
	A float64
	E float64
	C float64
}

type T3 struct {
	F float64
	G float64
	H float64
	I float64
	J float64
	K float64
	L float64
}

type T4 struct {
	M float64
	N float64
	O float64
	P float64
}

type T5 struct {
	Q float64
	R float64
	S float64
	T float64
	U float64
	V float64
}

type T6 struct {
	T9
	C T10
}

type T7 struct {
	T10
	T11
}

type T8 struct {
	T9
	C T7
}

type T9 struct {
	A T5
	B T3
	D T4
}

type T10 struct {
	W float64
}

type T11 struct {
	X float64
	Y float64
}

func MainTest(x T1, y T8, z T6) float64 {
	return Test(x.B, x.A, x.D, x.C, y.B, y.A, y.D, y.C, z.B, z.A, z.D,
		T7{
			T10: T10{
				W: z.C.W,
			},
			T11: T11{},
		},
	)
}
func Test(a T2, b T5, c T4, d T7, e T3, f T5, g T4, h T7, i T3, j T5, k T4, l T7) float64
