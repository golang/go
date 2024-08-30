// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type A interface {
	int | int64
}

type B interface {
	string
}

type C interface {
	String() string
}

type Myint int

func (i Myint) String() string {
	return "aa"
}

type T[P A, _ C, _ B] int

func (v T[P, Q, R]) test() {
	var r Q
	r.String()
}
