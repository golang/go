// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S[T comparable] struct {
	m map[T]T
}

func (s S[T]) M1(node T) {
	defer delete(s.m, node)
}

func (s S[T]) M2(node T) {
	defer func() {
		delete(s.m, node)
	}()
}

func (s S[T]) M3(node T) {
	defer f(s.m, node)
}

//go:noinline
func f[T comparable](map[T]T, T) {}

var _ = S[int]{}
