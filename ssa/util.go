// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines a number of miscellaneous utility functions.

import (
	"fmt"
	"go/ast"
	"io"
	"os"

	"code.google.com/p/go.tools/go/types"
)

func unreachable() {
	panic("unreachable")
}

//// AST utilities

// unparen returns e with any enclosing parentheses stripped.
func unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			break
		}
		e = p.X
	}
	return e
}

// isBlankIdent returns true iff e is an Ident with name "_".
// They have no associated types.Object, and thus no type.
//
func isBlankIdent(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	return ok && id.Name == "_"
}

//// Type utilities.  Some of these belong in go/types.

// isPointer returns true for types whose underlying type is a pointer.
func isPointer(typ types.Type) bool {
	_, ok := typ.Underlying().(*types.Pointer)
	return ok
}

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

// DefaultType returns the default "typed" type for an "untyped" type;
// it returns the incoming type for all other types.  The default type
// for untyped nil is untyped nil.
//
// Exported to exp/ssa/interp.
//
// TODO(gri): this is a copy of go/types.defaultType; export that function.
//
func DefaultType(typ types.Type) types.Type {
	if t, ok := typ.(*types.Basic); ok {
		k := t.Kind()
		switch k {
		case types.UntypedBool:
			k = types.Bool
		case types.UntypedInt:
			k = types.Int
		case types.UntypedRune:
			k = types.Rune
		case types.UntypedFloat:
			k = types.Float64
		case types.UntypedComplex:
			k = types.Complex128
		case types.UntypedString:
			k = types.String
		}
		typ = types.Typ[k]
	}
	return typ
}

// logStack prints the formatted "start" message to stderr and
// returns a closure that prints the corresponding "end" message.
// Call using 'defer logStack(...)()' to show builder stack on panic.
// Don't forget trailing parens!
//
func logStack(format string, args ...interface{}) func() {
	msg := fmt.Sprintf(format, args...)
	io.WriteString(os.Stderr, msg)
	io.WriteString(os.Stderr, "\n")
	return func() {
		io.WriteString(os.Stderr, msg)
		io.WriteString(os.Stderr, " end\n")
	}
}

// callsRecover reports whether f contains a direct call to recover().
func callsRecover(f *Function) bool {
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			if call, ok := instr.(*Call); ok {
				if blt, ok := call.Call.Value.(*Builtin); ok {
					if blt.Name() == "recover" {
						return true
					}
				}
			}
		}
	}
	return false
}
