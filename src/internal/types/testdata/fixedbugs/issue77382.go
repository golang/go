// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type T1[T any] [42]T2[T]

type T2[T any] [42]T

func _[T any]() {
	_ = unsafe.Sizeof(T1[T]{})
}
