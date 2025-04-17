// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[S []T, T any](S, T) {}

func _() {
	type L chan int
	f([]L{}, make(chan int))
	f([]L{}, make(L))
	f([]chan int{}, make(chan int))
	f /* ERROR "[]chan int does not satisfy []L ([]chan int missing in []p.L)" */ ([]chan int{}, make(L))
}

// test case from issue

func Append[S ~[]T, T any](s S, x ...T) S { /* implementation of append */ return s }

func _() {
        type MyPtr *int
        var x []MyPtr
        _ = append(x, new(int))
        _ = Append(x, new(int))
}
