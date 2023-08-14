// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T0 interface{
}

type T1 interface{
	~int
}

type T2 interface{
	comparable
}

type T3 interface {
	T0
	T1
	T2
}

func _() {
	_ = T0(0)
	_ = T1 /* ERROR "cannot use interface T1 in conversion" */ (1)
	_ = T2 /* ERROR "cannot use interface T2 in conversion" */ (2)
	_ = T3 /* ERROR "cannot use interface T3 in conversion" */ (3)
}
