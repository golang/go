// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func Bad() {
	m := make(map[int64]A)
	a := m[0]
	if len(a.B.C1.D2.E2.F1) != 0 ||
		len(a.B.C1.D2.E2.F2) != 0 ||
		len(a.B.C1.D2.E2.F3) != 0 ||
		len(a.B.C1.D2.E2.F4) != 0 ||
		len(a.B.C1.D2.E2.F5) != 0 ||
		len(a.B.C1.D2.E2.F6) != 0 ||
		len(a.B.C1.D2.E2.F7) != 0 ||
		len(a.B.C1.D2.E2.F8) != 0 ||
		len(a.B.C1.D2.E2.F9) != 0 ||
		len(a.B.C1.D2.E2.F10) != 0 ||
		len(a.B.C1.D2.E2.F11) != 0 ||
		len(a.B.C1.D2.E2.F16) != 0 {
		panic("bad")
	}
}

type A struct {
	B
}

type B struct {
	C1 C
	C2 C
}

type C struct {
	D1 D
	D2 D
}

type D struct {
	E1 E
	E2 E
	E3 E
	E4 E
}

type E struct {
	F1  string
	F2  string
	F3  string
	F4  string
	F5  string
	F6  string
	F7  string
	F8  string
	F9  string
	F10 string
	F11 string
	F12 string
	F13 string
	F14 string
	F15 string
	F16 string
}
