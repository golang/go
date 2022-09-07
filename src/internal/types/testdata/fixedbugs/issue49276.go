// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type S /* ERROR illegal cycle in declaration of S */ struct {
	_ [unsafe.Sizeof(s)]byte
}

var s S

// Since f is a pointer, this case could be valid.
// But it's pathological and not worth the expense.
type T struct {
	f *[unsafe.Sizeof(T /* ERROR illegal cycle in type declaration */ {})]int
}

// a mutually recursive case using unsafe.Sizeof
type (
	A1 struct {
		_ [unsafe.Sizeof(B1{})]int
	}

	B1 struct {
		_ [unsafe.Sizeof(A1 /* ERROR illegal cycle in type declaration */ {})]int
	}
)

// a mutually recursive case using len
type (
	A2 struct {
		f [len(B2{}.f)]int
	}

	B2 struct {
		f [len(A2 /* ERROR illegal cycle in type declaration */ {}.f)]int
	}
)

// test case from issue
type a struct {
	_ [42 - unsafe.Sizeof(a /* ERROR illegal cycle in type declaration */ {})]byte
}
