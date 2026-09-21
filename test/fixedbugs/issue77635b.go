// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 77635: test building values of zero-sized types.

package p

type T1 [2][0]int
type T2 [0][2]int
type T3 struct {
	t T1
	x *byte
}
type T4 struct {
	t T2
	x *byte
}

func f1(t T1) any {
	return t
}
func f2(t T2) any {
	return t
}
func f3(t T3) any {
	return t
}
func f4(t T4) any {
	return t
}
func f5(t T1) any {
	return T3{t:t}
}
func f6(t T2) any {
	return T4{t:t}
}
func f7(t T1) {
	use(T3{t:t})
}
func f8(t T2) {
	use(T4{t:t})
}

func g1(t T3, i int) {
	t.t[i][i] = 1
}
func g2(t T4, i int) {
	t.t[i][i] = 1
}
func g3(t *T3, i int) {
	t.t[i][i] = 1
}
func g4(t *T4, i int) {
	t.t[i][i] = 1
}

//go:noinline
func use(x any) {
}
