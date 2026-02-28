// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"cmd/compile/internal/base"
)

// Declaration stack & operations

// A dsym stores a symbol's shadowed declaration so that it can be
// restored once the block scope ends.
type dsym struct {
	sym *Sym // sym == nil indicates stack mark
	def Object
}

// dclstack maintains a stack of shadowed symbol declarations so that
// Popdcl can restore their declarations when a block scope ends.
var dclstack []dsym

// Pushdcl pushes the current declaration for symbol s (if any) so that
// it can be shadowed by a new declaration within a nested block scope.
func Pushdcl(s *Sym) {
	dclstack = append(dclstack, dsym{
		sym: s,
		def: s.Def,
	})
}

// Popdcl pops the innermost block scope and restores all symbol declarations
// to their previous state.
func Popdcl() {
	for i := len(dclstack); i > 0; i-- {
		d := &dclstack[i-1]
		s := d.sym
		if s == nil {
			// pop stack mark
			dclstack = dclstack[:i-1]
			return
		}

		s.Def = d.def

		// Clear dead pointer fields.
		d.sym = nil
		d.def = nil
	}
	base.Fatalf("popdcl: no stack mark")
}

// Markdcl records the start of a new block scope for declarations.
func Markdcl() {
	dclstack = append(dclstack, dsym{
		sym: nil, // stack mark
	})
}

func isDclstackValid() bool {
	for _, d := range dclstack {
		if d.sym == nil {
			return false
		}
	}
	return true
}

// PkgDef returns the definition associated with s at package scope.
func (s *Sym) PkgDef() Object {
	return *s.pkgDefPtr()
}

// SetPkgDef sets the definition associated with s at package scope.
func (s *Sym) SetPkgDef(n Object) {
	*s.pkgDefPtr() = n
}

func (s *Sym) pkgDefPtr() *Object {
	// Look for outermost saved declaration, which must be the
	// package scope definition, if present.
	for i := range dclstack {
		d := &dclstack[i]
		if s == d.sym {
			return &d.def
		}
	}

	// Otherwise, the declaration hasn't been shadowed within a
	// function scope.
	return &s.Def
}

func CheckDclstack() {
	if !isDclstackValid() {
		base.Fatalf("mark left on the dclstack")
	}
}
