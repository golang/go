// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[
        T0 any,
        T1 []int,
        T2 ~float64 | ~complex128 | chan int,
]() {
        _ = T0(nil /* ERROR "cannot convert nil to type T0" */ )
        _ = T1(1 /* ERRORx `cannot convert 1 .* to type T1` */ )
        _ = T2(2 /* ERRORx `cannot convert 2 .* to type T2` */ )
}

// test case from issue
func f[T interface{[]int}]() {
	_ = T(1 /* ERROR "cannot convert" */ )
}
