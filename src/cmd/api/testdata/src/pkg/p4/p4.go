// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p4

type Pair[T1 interface{ M() }, T2 ~int] struct {
	f1 T1
	f2 T2
}

func NewPair[T1 interface{ M() }, T2 ~int](v1 T1, v2 T2) Pair[T1, T2] {
	return Pair[T1, T2]{f1: v1, f2: v2}
}

func (p Pair[X1, _]) First() X1 {
	return p.f1
}

func (p Pair[_, X2]) Second() X2 {
	return p.f2
}

func Clone[S ~[]T, T any](s S) S {
	return append(S(nil), s...)
}
