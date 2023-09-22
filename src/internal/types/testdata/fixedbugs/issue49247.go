// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

func Add1024[T integer](s []T) {
	for i, v := range s {
		s[i] = v + 1024 // ERROR "cannot convert 1024 (untyped int constant) to type T"
	}
}

func f[T interface{ int8 }]() {
	println(T(1024 /* ERROR "cannot convert 1024 (untyped int value) to type T" */))
}
