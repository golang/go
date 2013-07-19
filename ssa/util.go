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

// namedTypeMethodIndex returns the method (and its index) named id
// within the set of explicitly declared concrete methods of named
// type typ.  If not found, panic ensues.
//
func namedTypeMethodIndex(typ *types.Named, id string) (int, *types.Func) {
	for i, n := 0, typ.NumMethods(); i < n; i++ {
		m := typ.Method(i)
		if m.Id() == id {
			return i, m
		}
	}
	panic(fmt.Sprint("method not found: ", id, " in named type ", typ))
}

// interfaceMethodIndex returns the method (and its index) named id
// within the method-set of interface type typ.  If not found, panic
// ensues.
//
func interfaceMethodIndex(typ *types.Interface, id string) (int, *types.Func) {
	for i, n := 0, typ.NumMethods(); i < n; i++ {
		m := typ.Method(i)
		if m.Id() == id {
			return i, m
		}
	}
	panic(fmt.Sprint("method not found: ", id, " in interface ", typ))
}

// isSuperinterface returns true if x is a superinterface of y,
// i.e.  x's methods are a subset of y's.
//
func isSuperinterface(x, y *types.Interface) bool {
	if y.NumMethods() < x.NumMethods() {
		return false
	}
	// TODO(adonovan): opt: this is quadratic.
outer:
	for i, n := 0, x.NumMethods(); i < n; i++ {
		xm := x.Method(i)
		for j, m := 0, y.NumMethods(); j < m; j++ {
			ym := y.Method(j)
			if xm.Id() == ym.Id() {
				if !types.IsIdentical(xm.Type(), ym.Type()) {
					return false // common name but conflicting types
				}
				continue outer
			}
		}
		return false // y doesn't have this method
	}
	return true
}

// canHaveConcreteMethods returns true iff typ may have concrete
// methods associated with it.  Callers must supply allowPtr=true.
//
// TODO(gri): consider putting this in go/types.  It's surprisingly subtle.
func canHaveConcreteMethods(typ types.Type, allowPtr bool) bool {
	switch typ := typ.(type) {
	case *types.Pointer:
		return allowPtr && canHaveConcreteMethods(typ.Elem(), false)
	case *types.Named:
		switch typ.Underlying().(type) {
		case *types.Pointer, *types.Interface:
			return false
		}
		return true
	case *types.Struct:
		return true
	}
	return false
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
