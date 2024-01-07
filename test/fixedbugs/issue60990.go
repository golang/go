// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct{ _, _ []int }

func F[_ int]() {
	var f0, f1 float64
	var b bool
	_ = func(T, float64) bool {
		b = deepEqual(0, 1)
		return func() bool {
			f1 = min(f0, 0)
			return b
		}()
	}(T{nil, nil}, min(0, f1))
	f0 = min(0, 1)
}

//go:noinline
func deepEqual(x, y any) bool {
	return x == y
}

func init() {
	F[int]()
}
