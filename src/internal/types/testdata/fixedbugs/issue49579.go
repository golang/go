// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type I[F any] interface {
	Q(*F)
}

func G[F any]() I[any] {
	return g /* ERRORx `cannot use g\[F\]{} .* as I\[any\] value in return statement: g\[F\] does not implement I\[any\] \(method Q has pointer receiver\)` */ [F]{}
}

type g[F any] struct{}

func (*g[F]) Q(*any) {}
