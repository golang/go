// run -gcflags=-G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {}

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

// TODO(danscales) enable once the compiler is fixed
// var _ = f0[Sf]
// var _ = f0t[Sf]

func f1[P interface {
	Sf
	m()
}](p P) {
	_ = p.f
	p.f = 0
	p.m()
}

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

// TODO(danscales) enable once the compiler is fixed
// var _ = f2[Sfm]

// special case: structural type is a named pointer type

type PSfm *Sfm

func f3[P interface{ PSfm }](p P) {
	_ = p.f
	p.f = 0
}

// TODO(danscales) enable once the compiler is fixed
// var _ = f3[PSfm]
