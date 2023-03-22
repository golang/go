// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

// ----------------------------------------------------------------------------
// Type-based Alias Anallysis
//
// Described in
// Amer Diwan, Kathryn S. McKinley, J. Eliot B. Moss: Type-Based Alias Analysis.
// PLDI 1998
//
// TBAA relies on the fact that Golang is a type-safe language, i.e. different
// pointer types cannot be converted to each other in Golang. Under assumption,
// TBAA attempts to identify whether two pointers may point to same memory based
// on their type and value semantics. They can be summarized as follows rules:
//
//	#0 unsafe pointer may aliases with anything even if their types are different
//	#1 a must aliases with b if a==b
//	#2 a.f aliases with b.g if f==g and a aliases with b
//	#3 a.f aliases with *b if they have same types
//	#4 a[i] aliases with *b if they have same types
//	#5 a.f never aliases with b[i]
//	#6 a[i] aliases with b[j] if a==b
//	#7 a aliases with b if they have same types
type AliasType uint

const (
	MustAlias AliasType = iota
	MayAlias
	NoAlias
)

func (at AliasType) String() string {
	switch at {
	case MustAlias:
		return fmt.Sprintf("MustAlias")
	case MayAlias:
		return fmt.Sprintf("MayAlias")
	case NoAlias:
		return fmt.Sprintf("NoAlias")
	}
	return fmt.Sprintf("Unknown")
}

func sameType(a, b *Value) bool {
	return a.Type == b.Type
}

func addressTaken(f *Func, a *Value) bool {
	// TODO: #3 and #4 could be better handled by checking whether the address
	// of a variable is taken.
	return true
}

// GetMemoryAlias check if two pointers may point to the same memory location.
func GetMemoryAlias(a, b *Value) AliasType {
	// #0 unsafe pointer may aliases with anything even if their types are different
	if a.Type.IsUnsafePtr() || b.Type.IsUnsafePtr() {
		return MayAlias
	}

	// #1 a must aliases with b if a==b
	if a == b {
		return MustAlias
	}

	// #2 a.f aliases with b.g if f==g and a aliases with b
	if a.Op == OpOffPtr && b.Op == OpOffPtr {
		off1 := a.AuxInt64()
		off2 := b.AuxInt64()
		if off1 == off2 {
			ptr1 := a.Args[0]
			ptr2 := b.Args[0]
			return GetMemoryAlias(ptr1, ptr2)
		} else {
			return NoAlias
		}
	}

	// #3 a.f aliases with *b if they have same types
	if a.Op == OpLoad && b.Op == OpOffPtr {
		if sameType(a.Args[0], b) {
			return MayAlias
		} else {
			return NoAlias
		}
	} else if b.Op == OpLoad && a.Op == OpOffPtr {
		if sameType(b.Args[0], a) {
			return MayAlias
		} else {
			return NoAlias
		}
	}

	// #4 a[i] aliases with *b if they have same types
	if a.Op == OpLoad && b.Op == OpPtrIndex {
		if sameType(a.Args[0], b) {
			return MayAlias
		} else {
			return NoAlias
		}
	} else if b.Op == OpLoad && a.Op == OpPtrIndex {
		if sameType(b.Args[0], a) {
			return MayAlias
		} else {
			return NoAlias
		}
	}

	// #5 a.f never aliases with b[i]
	if (a.Op == OpOffPtr && b.Op == OpPtrIndex) ||
		(b.Op == OpOffPtr && a.Op == OpPtrIndex) {
		return NoAlias
	}

	// #6 a[i] aliases with b[j] if a==b
	if a.Op == OpPtrIndex && b.Op == OpPtrIndex {
		at := GetMemoryAlias(a.Args[0], b.Args[0])
		// Note that two array access may alias even if i != j in complie time
		// because they may share the underlying slice.
		return at
	}

	// #7 a aliases with b if they have same types
	if !sameType(a, b) {
		return NoAlias
	}

	return MayAlias
}
