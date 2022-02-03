// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"constraints"
)

type Builder[T constraints.Integer] struct{}

func (r Builder[T]) New() T {
	return T(42)
}
