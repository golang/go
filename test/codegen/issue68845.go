// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type T1 struct {
	x string
}

func f1() *T1 {
	// amd64:-`MOVQ\s[$]0`,-`MOVUPS\sX15`
	return &T1{}
}

type T2 struct {
	x, y string
}

func f2() *T2 {
	// amd64:-`MOVQ\s[$]0`,-`MOVUPS\sX15`
	return &T2{}
}

type T3 struct {
	x complex128
}

func f3() *T3 {
	// amd64:-`MOVQ\s[$]0`,-`MOVUPS\sX15`
	return &T3{}
}

type T4 struct {
	x []byte
}

func f4() *T4 {
	// amd64:-`MOVQ\s[$]0`,-`MOVUPS\sX15`
	return &T4{}
}

type T5 struct {
	x any
}

func f5() *T5 {
	// amd64:-`MOVQ\s[$]0`,-`MOVUPS\sX15`
	return &T5{}
}
