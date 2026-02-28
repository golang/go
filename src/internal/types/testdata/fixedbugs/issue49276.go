// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type S /* ERROR "invalid recursive type S" */ struct {
	_ [unsafe.Sizeof(s)]byte
}

var s S

// Since f is a pointer, this case could be valid.
// But it's pathological and not worth the expense.
type T /* ERROR "invalid recursive type" */ struct {
	f *[unsafe.Sizeof(T{})]int
}

// a mutually recursive case using unsafe.Sizeof
type (
	A1/* ERROR "invalid recursive type" */ struct {
		_ [unsafe.Sizeof(B1{})]int
	}

	B1 struct {
		_ [unsafe.Sizeof(A1{})]int
	}
)

// a mutually recursive case using len
type (
	A2/* ERROR "invalid recursive type" */ struct {
		f [len(B2{}.f)]int
	}

	B2 struct {
		f [len(A2{}.f)]int
	}
)

// test case from issue
type a /* ERROR "invalid recursive type" */ struct {
	_ [42 - unsafe.Sizeof(a{})]byte
}
