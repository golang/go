// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {}

// Field accesses through type parameters are disabled
// until we have a more thorough understanding of the
// implications on the spec. See issue #51576.

/*
type Sf struct {
	f int
}

func f0[P Sf](p P) {
	_ = p.f
	p.f = 0
}

func f0t[P ~struct{ f int }](p P) {
	_ = p.f
	p.f = 0
}

var _ = f0[Sf]
var _ = f0t[Sf]

func f1[P interface {
	~struct{ f int }
	m()
}](p P) {
	_ = p.f
	p.f = 0
	p.m()
}

var _ = f1[Sfm]

type Sm struct{}

func (Sm) m() {}

type Sfm struct {
	f int
}

func (Sfm) m() {}

func f2[P interface {
	Sfm
	m()
}](p P) {
	_ = p.f
	p.f = 0
	p.m()
}

var _ = f2[Sfm]

// special case: core type is a named pointer type

type PSfm *Sfm

func f3[P interface{ PSfm }](p P) {
	_ = p.f
	p.f = 0
}

var _ = f3[PSfm]

// special case: core type is an unnamed pointer type

func f4[P interface{ *Sfm }](p P) {
	_ = p.f
	p.f = 0
}

var _ = f4[*Sfm]

type A int
type B int
type C float64

type Int interface {
	*Sf | A
	*Sf | B
}

func f5[P Int](p P) {
	_ = p.f
	p.f = 0
}

var _ = f5[*Sf]

type Int2 interface {
	*Sf | A
	any
	*Sf | C
}

func f6[P Int2](p P) {
	_ = p.f
	p.f = 0
}

var _ = f6[*Sf]

type Int3 interface {
	Sf
	~struct{ f int }
}

func f7[P Int3](p P) {
	_ = p.f
	p.f = 0
}

var _ = f7[Sf]

type Em1 interface {
	*Sf | A
}

type Em2 interface {
	*Sf | B
}

type Int4 interface {
	Em1
	Em2
	any
}

func f8[P Int4](p P) {
	_ = p.f
	p.f = 0
}

var _ = f8[*Sf]
*/
