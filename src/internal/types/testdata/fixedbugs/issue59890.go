// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() { g /* ERROR "cannot infer T" */ () }

func g[T any]() (_ /* ERROR "cannot use _ as value or type" */, int) { panic(0) }

// test case from issue

var _ = append(f /* ERROR "cannot infer T" */ ()())

func f[T any]() (_ /* ERROR "cannot use _" */, _ /* ERROR "cannot use _" */, int) {
	panic("not implemented")
}
