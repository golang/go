// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Ordered interface {
	~int | ~int64 | ~float64 | ~string
}

func Min[T Ordered](x, y T) T {
	if x < y {
		return x
	}
	return y
}
