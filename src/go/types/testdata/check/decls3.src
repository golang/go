// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// embedded types

package decls3

import "unsafe"
import "fmt"

// fields with the same name at the same level cancel each other out

func _() {
	type (
		T1 struct { X int }
		T2 struct { X int }
		T3 struct { T1; T2 } // X is embedded twice at the same level via T1->X, T2->X
	)

	var t T3
	_ = t.X /* ERROR "ambiguous selector t.X" */
}

func _() {
	type (
		T1 struct { X int }
		T2 struct { T1 }
		T3 struct { T1 }
		T4 struct { T2; T3 } // X is embedded twice at the same level via T2->T1->X, T3->T1->X
	)

	var t T4
	_ = t.X /* ERROR "ambiguous selector t.X" */
}

func issue4355() {
	type (
	    T1 struct {X int}
	    T2 struct {T1}
	    T3 struct {T2}
	    T4 struct {T2}
	    T5 struct {T3; T4} // X is embedded twice at the same level via T3->T2->T1->X, T4->T2->T1->X
	)

	var t T5
	_ = t.X /* ERROR "ambiguous selector t.X" */
}

func _() {
	type State int
	type A struct{ State }
	type B struct{ fmt.State }
	type T struct{ A; B }

	var t T
	_ = t.State /* ERROR "ambiguous selector t.State" */
}

// Embedded fields can be predeclared types.

func _() {
	type T0 struct{
		int
		float32
		f int
	}
	var x T0
	_ = x.int
	_ = x.float32
	_ = x.f

	type T1 struct{
		T0
	}
	var y T1
	_ = y.int
	_ = y.float32
	_ = y.f
}

// Restrictions on embedded field types.

func _() {
	type I1 interface{}
	type I2 interface{}
	type P1 *int
	type P2 *int
	type UP unsafe.Pointer

	type T1 struct {
		I1
		* /* ERROR "cannot be a pointer to an interface" */ I2
		* /* ERROR "cannot be a pointer to an interface" */ error
		P1 /* ERROR "cannot be a pointer" */
		* /* ERROR "cannot be a pointer" */ P2
	}

	// unsafe.Pointers are treated like regular pointers when embedded
	type T2 struct {
		unsafe /* ERROR "cannot be unsafe.Pointer" */ .Pointer
		*/* ERROR "cannot be unsafe.Pointer" */ /* ERROR "Pointer redeclared" */ unsafe.Pointer
		UP /* ERROR "cannot be unsafe.Pointer" */
		* /* ERROR "cannot be unsafe.Pointer" */  /* ERROR "UP redeclared" */ UP
	}
}

// Named types that are pointers.

type S struct{ x int }
func (*S) m() {}
type P *S

func _() {
	var s *S
	_ = s.x
	_ = s.m

	var p P
	_ = p.x
	_ = p.m /* ERROR "no field or method" */
	_ = P.m /* ERROR "no field or method" */
}

// Borrowed from the FieldByName test cases in reflect/all_test.go.

type D1 struct {
	d int
}
type D2 struct {
	d int
}

type S0 struct {
	A, B, C int
	D1
	D2
}

type S1 struct {
	B int
	S0
}

type S2 struct {
	A int
	*S1
}

type S1x struct {
	S1
}

type S1y struct {
	S1
}

type S3 struct {
	S1x
	S2
	D, E int
	*S1y
}

type S4 struct {
	*S4
	A int
}

// The X in S6 and S7 annihilate, but they also block the X in S8.S9.
type S5 struct {
	S6
	S7
	S8
}

type S6 struct {
	X int
}

type S7 S6

type S8 struct {
	S9
}

type S9 struct {
	X int
	Y int
}

// The X in S11.S6 and S12.S6 annihilate, but they also block the X in S13.S8.S9.
type S10 struct {
	S11
	S12
	S13
}

type S11 struct {
	S6
}

type S12 struct {
	S6
}

type S13 struct {
	S8
}

func _() {
	_ = struct{}{}.Foo /* ERROR "no field or method" */
	_ = S0{}.A
	_ = S0{}.D /* ERROR "no field or method" */
	_ = S1{}.A
	_ = S1{}.B
	_ = S1{}.S0
	_ = S1{}.C
	_ = S2{}.A
	_ = S2{}.S1
	_ = S2{}.B
	_ = S2{}.C
	_ = S2{}.D /* ERROR "no field or method" */
	_ = S3{}.S1 /* ERROR "ambiguous selector \(S3 literal\).S1" */
	_ = S3{}.A
	_ = S3{}.B /* ERROR "ambiguous selector" \(S3 literal\).B */
	_ = S3{}.D
	_ = S3{}.E
	_ = S4{}.A
	_ = S4{}.B /* ERROR "no field or method" */
	_ = S5{}.X /* ERROR "ambiguous selector \(S5 literal\).X" */
	_ = S5{}.Y
	_ = S10{}.X /* ERROR "ambiguous selector \(S10 literal\).X" */
	_ = S10{}.Y
}

// Borrowed from the FieldByName benchmark in reflect/all_test.go.

type R0 struct {
	*R1
	*R2
	*R3
	*R4
}

type R1 struct {
	*R5
	*R6
	*R7
	*R8
}

type R2 R1
type R3 R1
type R4 R1

type R5 struct {
	*R9
	*R10
	*R11
	*R12
}

type R6 R5
type R7 R5
type R8 R5

type R9 struct {
	*R13
	*R14
	*R15
	*R16
}

type R10 R9
type R11 R9
type R12 R9

type R13 struct {
	*R17
	*R18
	*R19
	*R20
}

type R14 R13
type R15 R13
type R16 R13

type R17 struct {
	*R21
	*R22
	*R23
	*R24
}

type R18 R17
type R19 R17
type R20 R17

type R21 struct {
	X int
}

type R22 R21
type R23 R21
type R24 R21

var _ = R0{}.X /* ERROR "ambiguous selector \(R0 literal\).X" */