// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Optional[T any] struct {}

func (_ Optional[T]) Val() (T, bool)

type Box[T any] interface {
	Val() (T, bool)
}

func f[V interface{}, A, B Box[V]]() {}

func _() {
	f[int, Optional[int], Optional[int]]()
	_ = f[int, Optional[int], Optional /* ERROR does not implement Box */ [string]]
	_ = f[int, Optional[int], Optional /* ERROR Optional.* does not implement Box.* */ [string]]
}
