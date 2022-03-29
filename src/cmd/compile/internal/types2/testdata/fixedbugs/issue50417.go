// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Field accesses through type parameters are disabled
// until we have a more thorough understanding of the
// implications on the spec. See issue #51576.

package p

type Sf struct {
        f int
}

func f0[P Sf](p P) {
        _ = p.f // ERROR p\.f undefined
        p.f /* ERROR p\.f undefined */ = 0
}

func f0t[P ~struct{f int}](p P) {
        _ = p.f // ERROR p\.f undefined
        p.f /* ERROR p\.f undefined */ = 0
}

var _ = f0[Sf]
var _ = f0t[Sf]

var _ = f0[Sm /* ERROR does not implement */ ]
var _ = f0t[Sm /* ERROR does not implement */ ]

func f1[P interface{ Sf; m() }](p P) {
        _ = p.f // ERROR p\.f undefined
        p.f /* ERROR p\.f undefined */ = 0
        p.m()
}

var _ = f1[Sf /* ERROR missing method m */ ]
var _ = f1[Sm /* ERROR does not implement */ ]

type Sm struct {}

func (Sm) m() {}

type Sfm struct {
        f int
}

func (Sfm) m() {}

func f2[P interface{ Sfm; m() }](p P) {
        _ = p.f // ERROR p\.f undefined
        p.f /* ERROR p\.f undefined */ = 0
        p.m()
}

var _ = f2[Sfm]

// special case: core type is a named pointer type

type PSfm *Sfm

func f3[P interface{ PSfm }](p P) {
        _ = p.f // ERROR p\.f undefined
        p.f /* ERROR p\.f undefined */ = 0
        p.m /* ERROR type P has no field or method m */ ()
}

var _ = f3[PSfm]
