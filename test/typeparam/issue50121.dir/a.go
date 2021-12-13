// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"constraints"
	"math/rand"
)

type Builder[T constraints.Integer] struct{}

func (r Builder[T]) New() T {
	return T(rand.Int())
}

var IntBuilder = Builder[int]{}

func BuildInt() int {
	return IntBuilder.New()
}
