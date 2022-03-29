// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[_ any] int

func f[_ any]() {}
func g[_, _ any]() {}

func _() {
	_ = f[T /* ERROR without instantiation */ ]
	_ = g[T /* ERROR without instantiation */ , T /* ERROR without instantiation */ ]
}