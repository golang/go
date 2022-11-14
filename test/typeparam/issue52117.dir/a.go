// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func Compare[T int | uint](a, b T) int {
	return 0
}

type Slice[T int | uint] struct{}

func (l Slice[T]) Comparator() func(v1, v2 T) int {
	return Compare[T]
}
