// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func Fact[T interface{ int | int64 | float64 }](n T) T {
	if n == 1 {
		return 1
	}
	return n * Fact(n-1)
}
