// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Comparator[T any] func(v1, v2 T) int

func CompareInt[T ~int](a, b T) int {
	if a < b {
		return -1
	}
	if a == b {
		return 0
	}
	return 1
}
