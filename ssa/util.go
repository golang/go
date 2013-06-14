package ssa

// This file defines a number of miscellaneous utility functions.

import (
	"fmt"
	"go/ast"
	"io"
	"os"
	"reflect"

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
// TODO(gri): consider making typechecker not treat them differently.
// It's one less thing for clients like us to worry about.
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

// pointer(typ) returns the type that is a pointer to typ.
// TODO(adonovan): inline and eliminate.
func pointer(typ types.Type) *types.Pointer {
	return types.NewPointer(typ)
}

// namedTypeMethodIndex returns the method (and its index) named id
// within the set of explicitly declared concrete methods of named
// type typ.  If not found, panic ensues.
//
// TODO(gri): move this functionality into the go/types API?
//
func namedTypeMethodIndex(typ *types.Named, id Id) (int, *types.Func) {
	for i, n := 0, typ.NumMethods(); i < n; i++ {
		m := typ.Method(i)
		if MakeId(m.Name(), m.Pkg()) == id {
			return i, m
		}
	}
	panic(fmt.Sprint("method not found: ", id, " in named type ", typ))
}

// interfaceMethodIndex returns the method (and its index) named id
// within the method-set of interface type typ.  If not found, panic
// ensues.
//
// TODO(gri): move this functionality into the go/types API.
//
func interfaceMethodIndex(typ *types.Interface, id Id) (int, *types.Func) {
	for i, n := 0, typ.NumMethods(); i < n; i++ {
		m := typ.Method(i)
		if MakeId(m.Name(), m.Pkg()) == id {
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
			if MakeId(xm.Name(), xm.Pkg()) == MakeId(ym.Name(), ym.Pkg()) {
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
// it returns the incoming type for all other types. If there is no
// corresponding untyped type, the result is types.Typ[types.Invalid].
//
// Exported to exp/ssa/interp.
//
// TODO(gri): this is a copy of go/types.defaultType; export that function.
//
func DefaultType(typ types.Type) types.Type {
	if t, ok := typ.(*types.Basic); ok {
		k := types.Invalid
		switch t.Kind() {
		// case UntypedNil:
		//      There is no default type for nil. For a good error message,
		//      catch this case before calling this function.
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

// makeId returns the Id (name, pkg) if the name is exported or
// (name, nil) otherwise.
//
// Exported to exp/ssa/interp.
//
func MakeId(name string, pkg *types.Package) (id Id) {
	id.Name = name
	if !ast.IsExported(name) {
		id.Pkg = pkg
		// TODO(gri): fix
		// if pkg.Path() == "" {
		// 	panic("Package " + pkg.Name() + "has empty Path")
		// }
	}
	return
}

type ids []Id // a sortable slice of Id

func (p ids) Len() int { return len(p) }
func (p ids) Less(i, j int) bool {
	x, y := p[i], p[j]
	// *Package pointers are canonical so order by them.
	// Don't use x.Pkg.ImportPath because sometimes it's empty.
	// (TODO(gri): fix that.)
	return reflect.ValueOf(x.Pkg).Pointer() < reflect.ValueOf(y.Pkg).Pointer() ||
		x.Pkg == y.Pkg && x.Name < y.Name
}
func (p ids) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

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
