// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "cmd/compile/internal/base"

// AlgKind describes the kind of algorithms used for comparing and
// hashing a Type.
type AlgKind int8

//go:generate stringer -type AlgKind -trimprefix A alg.go

const (
	AUNK   AlgKind = iota
	ANOEQ          // Types cannot be compared
	ANOALG         // implies ANOEQ, and in addition has a part that is marked Noalg
	AMEM           // Type can be compared/hashed as regular memory.
	AMEM0          // Specific subvariants of AMEM (TODO: move to ../reflectdata?)
	AMEM8
	AMEM16
	AMEM32
	AMEM64
	AMEM128
	ASTRING
	AINTER
	ANILINTER
	AFLOAT32
	AFLOAT64
	ACPLX64
	ACPLX128
	ASPECIAL // Type needs special comparison/hashing functions.
)

// Most kinds are priority 0. Higher numbers are higher priority, in that
// the higher priority kinds override lower priority kinds.
var algPriority = [ASPECIAL + 1]int8{ASPECIAL: 1, ANOEQ: 2, ANOALG: 3, AMEM: -1}

// setAlg sets the algorithm type of t to a, if it is of higher
// priority to the current algorithm type.
func (t *Type) setAlg(a AlgKind) {
	if t.alg == AUNK {
		base.Fatalf("setAlg(%v,%s) starting with unknown priority", t, a)
	}
	if algPriority[a] > algPriority[t.alg] {
		t.alg = a
	} else if a != t.alg && algPriority[a] == algPriority[t.alg] {
		base.Fatalf("ambiguous priority %s and %s", a, t.alg)
	}
}

// AlgType returns the AlgKind used for comparing and hashing Type t.
func AlgType(t *Type) AlgKind {
	CalcSize(t)
	return t.alg
}

// TypeHasNoAlg reports whether t does not have any associated hash/eq
// algorithms because t, or some component of t, is marked Noalg.
func TypeHasNoAlg(t *Type) bool {
	return AlgType(t) == ANOALG
}

// IsComparable reports whether t is a comparable type.
func IsComparable(t *Type) bool {
	a := AlgType(t)
	return a != ANOEQ && a != ANOALG
}

// IncomparableField returns an incomparable Field of struct Type t, if any.
func IncomparableField(t *Type) *Field {
	for _, f := range t.Fields() {
		if !IsComparable(f.Type) {
			return f
		}
	}
	return nil
}

// IsPaddedField reports whether the i'th field of struct type t is followed
// by padding.
func IsPaddedField(t *Type, i int) bool {
	if !t.IsStruct() {
		base.Fatalf("IsPaddedField called non-struct %v", t)
	}
	end := t.width
	if i+1 < t.NumFields() {
		end = t.Field(i + 1).Offset
	}
	return t.Field(i).End() != end
}
