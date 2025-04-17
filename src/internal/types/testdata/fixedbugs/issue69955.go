// -gotypesalias=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "math/big"

type (
	S struct{}
	N int

	A = S
	B = int
	C = N
)

var (
	i int
	s S
	n N
	a A
	b B
	c C
	w big.Word
)

const (
	_ = i // ERROR "i (variable of type int) is not constant"
	_ = s // ERROR "s (variable of struct type S) is not constant"
	_ = struct /* ERROR "struct{}{} (value of type struct{}) is not constant" */ {}{}
	_ = n // ERROR "n (variable of int type N) is not constant"

	_ = a // ERROR "a (variable of struct type A) is not constant"
	_ = b // ERROR "b (variable of int type B) is not constant"
	_ = c // ERROR "c (variable of int type C) is not constant"
	_ = w // ERROR "w (variable of uint type big.Word) is not constant"
)

var _ int = w /* ERROR "cannot use w + 1 (value of uint type big.Word) as int value in variable declaration" */ + 1
