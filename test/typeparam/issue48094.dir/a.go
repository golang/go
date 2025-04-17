// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "unsafe"

func F[T any]() uintptr {
	var t T
	return unsafe.Sizeof(t)
}

func G[T any]() uintptr {
	var t T
	return unsafe.Alignof(t)
}

//func H[T any]() uintptr {
//	type S struct {
//		a T
//		b T
//	}
//	var s S
//	return unsafe.Offsetof(s.b)
//}
