// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	. "go/types"
)

// checkInvariants verifies structural well-formedness and consistency of
// a type-checked package and its recorded type information:
//  1. No Object has a nil Type().
//  2. Every *Func object dynamically has a *Signature type.
//  3. No compound type component (Pointer.Elem, Slice.Elem, Array.Elem,
//     Map.Key, Map.Elem, Chan.Elem, Struct.Field, Tuple.At, Interface.Method,
//     Interface.EmbeddedType, TypeParam.Constraint, Union.Term) is nil
//     (empty Signature.Params and Results are canonically nil *Tuples).
//  4. Accessor methods (Underlying, Rhs, Constraint, Unalias) never panic.
//  5. Typ[Invalid] is never published for typed objects (*Var, *Const, *TypeName,
//     *Func) or non-builtin expression types unless the package reported an error.
//     (Non-typed objects like *PkgName, *Builtin, and *Label, and built-in
//     expressions, have Typ[Invalid] by specification).
func checkInvariants(pkg *Package, info *Info, hasErrors bool) []error {
	v := &validator{
		pkg:       pkg,
		info:      info,
		hasErrors: hasErrors,
		seenTypes: make(map[Type]bool),
		seenObjs:  make(map[Object]bool),
	}
	if pkg != nil {
		v.checkScope(pkg.Scope(), "pkg.Scope()")
	}
	v.checkInfo(info)
	return v.errs
}

type validator struct {
	pkg       *Package
	info      *Info
	hasErrors bool
	seenTypes map[Type]bool
	seenObjs  map[Object]bool
	errs      []error
}

func (v *validator) errorf(format string, args ...any) {
	v.errs = append(v.errs, fmt.Errorf(format, args...))
}

func (v *validator) checkScope(s *Scope, where string) {
	if s == nil {
		return
	}
	for _, name := range s.Names() {
		obj := s.Lookup(name)
		v.checkObject(obj, fmt.Sprintf("%s.Lookup(%q)", where, name))
	}
	for i := 0; i < s.NumChildren(); i++ {
		v.checkScope(s.Child(i), fmt.Sprintf("%s.Child(%d)", where, i))
	}
}

func (v *validator) checkInfo(info *Info) {
	if info == nil {
		return
	}
	for ident, obj := range info.Defs {
		if obj != nil {
			v.checkObject(obj, fmt.Sprintf("Info.Defs[%s]", ident.Name))
		}
	}
	for ident, obj := range info.Uses {
		if obj != nil {
			v.checkObject(obj, fmt.Sprintf("Info.Uses[%s]", ident.Name))
		}
	}
	for node, obj := range info.Implicits {
		if obj != nil {
			v.checkObject(obj, fmt.Sprintf("Info.Implicits[%T]", node))
		}
	}
	for _, s := range info.Scopes {
		v.checkScope(s, "Info.Scopes")
	}
	for expr, tv := range info.Types {
		if tv.Type == nil {
			v.errorf("Info.Types[%v] has nil Type", expr)
		} else if !tv.IsBuiltin() {
			v.checkType(tv.Type, fmt.Sprintf("Info.Types[%v]", expr))
		}
	}
}

func (v *validator) checkObject(obj Object, where string) {
	if obj == nil || v.seenObjs[obj] {
		return
	}
	v.seenObjs[obj] = true

	// Invariant 1: obj.Type() must never be nil.
	typ := obj.Type()
	if typ == nil {
		v.errorf("object %q (%T) at %v has nil Type() in %s", obj.Name(), obj, obj.Pos(), where)
		return
	}

	// Invariant 2: Func.Type() must dynamically be a *Signature.
	if fn, ok := obj.(*Func); ok {
		if _, ok := typ.(*Signature); !ok {
			v.errorf("func %s has type %T, want *Signature in %s", fn.Name(), typ, where)
		}
	}

	// PkgName, Builtin, and Label don't have a Go type; their Type()
	// is defined as Typ[Invalid] by specification.
	switch obj.(type) {
	case *PkgName, *Builtin, *Label:
		return
	}

	v.checkType(typ, fmt.Sprintf("object %s", obj.Name()))
}

func (v *validator) checkType(typ Type, where string) {
	if typ == nil {
		v.errorf("nil Type encountered at %s", where)
		return
	}
	if v.seenTypes[typ] {
		return
	}
	v.seenTypes[typ] = true

	// Safe Underlying() call: must never panic.
	var under Type
	func() {
		defer func() {
			if r := recover(); r != nil {
				v.errorf("panic calling Underlying() on %T at %s: %v", typ, where, r)
			}
		}()
		under = typ.Underlying()
	}()
	if under == nil {
		v.errorf("type %T at %s has nil Underlying()", typ, where)
	}

	switch t := typ.(type) {
	case *Basic:
		// Invariant: Typ[Invalid] is never published without a reported error.
		if t == Typ[Invalid] && !v.hasErrors {
			v.errorf("Typ[Invalid] published at %s without reported package errors", where)
		}

	case *Pointer:
		if t.Elem() == nil {
			v.errorf("Pointer at %s has nil Elem()", where)
		} else {
			v.checkType(t.Elem(), where+".Elem()")
		}

	case *Array:
		if t.Elem() == nil {
			v.errorf("Array at %s has nil Elem()", where)
		} else {
			v.checkType(t.Elem(), where+".Elem()")
		}

	case *Slice:
		if t.Elem() == nil {
			v.errorf("Slice at %s has nil Elem()", where)
		} else {
			v.checkType(t.Elem(), where+".Elem()")
		}

	case *Map:
		if t.Key() == nil {
			v.errorf("Map at %s has nil Key()", where)
		} else {
			v.checkType(t.Key(), where+".Key()")
		}
		if t.Elem() == nil {
			v.errorf("Map at %s has nil Elem()", where)
		} else {
			v.checkType(t.Elem(), where+".Elem()")
		}

	case *Chan:
		if t.Elem() == nil {
			v.errorf("Chan at %s has nil Elem()", where)
		} else {
			v.checkType(t.Elem(), where+".Elem()")
		}

	case *Struct:
		for i := 0; i < t.NumFields(); i++ {
			f := t.Field(i)
			if f == nil {
				v.errorf("Struct at %s has nil Field(%d)", where, i)
			} else {
				v.checkObject(f, fmt.Sprintf("%s.Field(%d:%s)", where, i, f.Name()))
			}
		}

	case *Tuple:
		for i := 0; i < t.Len(); i++ {
			elem := t.At(i)
			if elem == nil {
				v.errorf("Tuple at %s has nil At(%d)", where, i)
			} else {
				v.checkObject(elem, fmt.Sprintf("%s.At(%d)", where, i))
			}
		}

	case *Signature:
		if recv := t.Recv(); recv != nil {
			v.checkObject(recv, where+".Recv()")
		}
		if params := t.Params(); params != nil {
			v.checkType(params, where+".Params()")
		}
		if results := t.Results(); results != nil {
			v.checkType(results, where+".Results()")
		}
		v.checkTypeParamList(t.TypeParams(), where+".TypeParams()")
		v.checkTypeParamList(t.RecvTypeParams(), where+".RecvTypeParams()")

	case *Interface:
		for i := 0; i < t.NumMethods(); i++ {
			m := t.Method(i)
			if m == nil {
				v.errorf("Interface at %s has nil Method(%d)", where, i)
			} else {
				v.checkObject(m, fmt.Sprintf("%s.Method(%s)", where, m.Name()))
			}
		}
		for i := 0; i < t.NumEmbeddeds(); i++ {
			emb := t.EmbeddedType(i)
			if emb == nil {
				v.errorf("Interface at %s has nil EmbeddedType(%d)", where, i)
			} else {
				v.checkType(emb, fmt.Sprintf("%s.EmbeddedType(%d)", where, i))
			}
		}

	case *Named:
		if t.Obj() == nil {
			v.errorf("Named at %s has nil Obj()", where)
		} else {
			v.checkObject(t.Obj(), where+".Obj()")
		}
		if targs := t.TypeArgs(); targs != nil {
			for i := 0; i < targs.Len(); i++ {
				v.checkType(targs.At(i), fmt.Sprintf("%s.TypeArgs(%d)", where, i))
			}
		}
		// Only check methods on origin types. On instantiated types (t.Origin() != t),
		// methods are lazily expanded from the origin; eagerly traversing methods of
		// instantiated types whose methods return further generic instances (e.g.
		// func (List[E]) Map[F any](...) List[F]) leads to infinite recursive method
		// instantiation.
		if t.Origin() == t {
			for i := 0; i < t.NumMethods(); i++ {
				m := t.Method(i)
				if m == nil {
					v.errorf("Named at %s has nil Method(%d)", where, i)
				} else {
					v.checkObject(m, fmt.Sprintf("%s.Method(%s)", where, m.Name()))
				}
			}
		}

	case *Alias:
		func() {
			defer func() {
				if r := recover(); r != nil {
					v.errorf("panic calling Rhs() on Alias at %s: %v", where, r)
				}
			}()
			rhs := t.Rhs()
			if rhs == nil {
				v.errorf("Alias at %s has nil Rhs()", where)
			} else {
				v.checkType(rhs, where+".Rhs()")
			}
		}()
		func() {
			defer func() {
				if r := recover(); r != nil {
					v.errorf("panic calling Unalias() on Alias at %s: %v", where, r)
				}
			}()
			if actual := Unalias(t); actual == nil {
				v.errorf("Unalias() returned nil on Alias at %s", where)
			}
		}()

	case *TypeParam:
		func() {
			defer func() {
				if r := recover(); r != nil {
					v.errorf("panic calling Constraint() on TypeParam at %s: %v", where, r)
				}
			}()
			c := t.Constraint()
			if c == nil {
				v.errorf("TypeParam at %s has nil Constraint()", where)
			} else {
				v.checkType(c, where+".Constraint()")
			}
		}()
		if t.Obj() == nil {
			v.errorf("TypeParam at %s has nil Obj()", where)
		} else {
			v.checkObject(t.Obj(), where+".Obj()")
		}

	case *Union:
		for i := 0; i < t.Len(); i++ {
			term := t.Term(i)
			if term == nil || term.Type() == nil {
				v.errorf("Union at %s has nil term at %d", where, i)
			} else {
				v.checkType(term.Type(), fmt.Sprintf("%s.Term(%d)", where, i))
			}
		}
	}
}

func (v *validator) checkTypeParamList(tparams *TypeParamList, where string) {
	if tparams == nil {
		return
	}
	for i := 0; i < tparams.Len(); i++ {
		tp := tparams.At(i)
		if tp == nil {
			v.errorf("%s at index %d is nil", where, i)
		} else {
			v.checkType(tp, fmt.Sprintf("%s.At(%d)", where, i))
		}
	}
}

func TestBrokenCodeInvariants(t *testing.T) {
	tests := []struct {
		name string
		src  string
	}{
		{
			name: "cyclic aliases",
			src: `package p
				type A = B
				type B = A`,
		},
		{
			name: "cyclic defined types",
			src: `package p
				type A B
				type B A`,
		},
		{
			name: "mutually recursive alias and struct",
			src: `package p
				type A = struct { b B }
				type B = struct { a A }`,
		},
		{
			name: "broken method receiver",
			src: `package p
				func (r Undeclared) M() {}`,
		},
		{
			name: "generic receiver with wrong param count",
			src: `package p
				type T[A, B any] struct{}
				func (r T[A]) M() {}`,
		},
		{
			name: "broken signature parameters and results",
			src: `package p
				func f(a undeclared) (b undeclared) { return }`,
		},
		{
			name: "broken short var decl in func body",
			src: `package p
				func f() {
					x := undeclared
					_ = x
				}`,
		},
		{
			name: "broken struct fields",
			src: `package p
				type S struct {
					f undeclared
				}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "test.go", tt.src, 0)
			if err != nil {
				t.Fatal(err)
			}
			conf := Config{
				Error: func(error) {}, // ignore expected errors
			}
			info := Info{
				Types:     make(map[ast.Expr]TypeAndValue),
				Defs:      make(map[*ast.Ident]Object),
				Uses:      make(map[*ast.Ident]Object),
				Implicits: make(map[ast.Node]Object),
				Scopes:    make(map[ast.Node]*Scope),
			}
			pkg, _ := conf.Check("p", fset, []*ast.File{f}, &info)
			if errs := checkInvariants(pkg, &info, true); len(errs) > 0 {
				for _, err := range errs {
					t.Errorf("invariant violated: %v", err)
				}
			}
		})
	}
}
