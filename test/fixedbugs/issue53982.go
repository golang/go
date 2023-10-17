// build

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S[K, V any] struct {
	E[V]
}

type E[K any] struct{}

func (e E[K]) M() E[K] {
	return e
}

func G[K, V any](V) {
	_ = (*S[K, V]).M
}

func main() {
	G[*int](new(int))
}
