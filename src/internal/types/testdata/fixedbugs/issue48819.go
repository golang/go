// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type T /* ERROR "invalid recursive type: T refers to itself" */ struct {
	T
}

func _(t T) {
	_ = unsafe.Sizeof(t) // should not go into infinite recursion here
}
