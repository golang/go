// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that we get an empty type set (not just an error)
// when using an invalid ~A.

package p

type A int
type C interface {
	~ /* ERROR "invalid use of ~" */ A
}

func f[_ C]()              {}
func g[_ interface{ C }]() {}
func h[_ C | int]()        {}

func _() {
	_ = f[int /* ERROR "cannot satisfy C (empty type set)" */]
	_ = g[int /* ERROR "cannot satisfy interface{C} (empty type set)" */]
	_ = h[int]
}
