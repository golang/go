// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"go/ast"
	"go/token"
	"log"
)


/*
 * Type compiler
 */

type typeCompiler struct {
	*compiler
	block *block
	// Check to be performed after a type declaration is compiled.
	//
	// TODO(austin) This will probably have to change after we
	// eliminate forward declarations.
	lateCheck func() bool
}

func (a *typeCompiler) compileIdent(x *ast.Ident, allowRec bool) Type {
	_, _, def := a.block.Lookup(x.Value)
	if def == nil {
		a.diagAt(x, "%s: undefined", x.Value)
		return nil
	}
	switch def := def.(type) {
	case *Constant:
		a.diagAt(x, "constant %v used as type", x.Value)
		return nil
	case *Variable:
		a.diagAt(x, "variable %v used as type", x.Value)
		return nil
	case *NamedType:
		if !allowRec && def.incomplete {
			a.diagAt(x, "illegal recursive type")
			return nil
		}
		if !def.incomplete && def.Def == nil {
			// Placeholder type from an earlier error
			return nil
		}
		return def
	case Type:
		return def
	}
	log.Crashf("name %s has unknown type %T", x.Value, def)
	return nil
}

func (a *typeCompiler) compileArrayType(x *ast.ArrayType, allowRec bool) Type {
	// Compile element type
	elem := a.compileType(x.Elt, allowRec)

	// Compile length expression
	if x.Len == nil {
		if elem == nil {
			return nil
		}
		return NewSliceType(elem)
	}

	if _, ok := x.Len.(*ast.Ellipsis); ok {
		a.diagAt(x.Len, "... array initailizers not implemented")
		return nil
	}
	l, ok := a.compileArrayLen(a.block, x.Len)
	if !ok {
		return nil
	}
	if l < 0 {
		a.diagAt(x.Len, "array length must be non-negative")
		return nil
	}
	if elem == nil {
		return nil
	}

	return NewArrayType(l, elem)
}

func countFields(fs []*ast.Field) int {
	n := 0
	for _, f := range fs {
		if f.Names == nil {
			n++
		} else {
			n += len(f.Names)
		}
	}
	return n
}

func (a *typeCompiler) compileFields(fs []*ast.Field, allowRec bool) ([]Type, []*ast.Ident, []token.Position, bool) {
	n := countFields(fs)
	ts := make([]Type, n)
	ns := make([]*ast.Ident, n)
	ps := make([]token.Position, n)

	bad := false
	i := 0
	for _, f := range fs {
		t := a.compileType(f.Type, allowRec)
		if t == nil {
			bad = true
		}
		if f.Names == nil {
			ns[i] = nil
			ts[i] = t
			ps[i] = f.Type.Pos()
			i++
			continue
		}
		for _, n := range f.Names {
			ns[i] = n
			ts[i] = t
			ps[i] = n.Pos()
			i++
		}
	}

	return ts, ns, ps, bad
}

func (a *typeCompiler) compileStructType(x *ast.StructType, allowRec bool) Type {
	ts, names, poss, bad := a.compileFields(x.Fields, allowRec)

	// XXX(Spec) The spec claims that field identifiers must be
	// unique, but 6g only checks this when they are accessed.  I
	// think the spec is better in this regard: if I write two
	// fields with the same name in the same struct type, clearly
	// that's a mistake.  This definition does *not* descend into
	// anonymous fields, so it doesn't matter if those change.
	// There's separate language in the spec about checking
	// uniqueness of field names inherited from anonymous fields
	// at use time.
	fields := make([]StructField, len(ts))
	nameSet := make(map[string]token.Position, len(ts))
	for i := range fields {
		// Compute field name and check anonymous fields
		var name string
		if names[i] != nil {
			name = names[i].Value
		} else {
			if ts[i] == nil {
				continue
			}

			var nt *NamedType
			// [For anonymous fields,] the unqualified
			// type name acts as the field identifier.
			switch t := ts[i].(type) {
			case *NamedType:
				name = t.Name
				nt = t
			case *PtrType:
				switch t := t.Elem.(type) {
				case *NamedType:
					name = t.Name
					nt = t
				}
			}
			// [An anonymous field] must be specified as a
			// type name T or as a pointer to a type name
			// *T, and T itself, may not be a pointer or
			// interface type.
			if nt == nil {
				a.diagAt(&poss[i], "embedded type must T or *T, where T is a named type")
				bad = true
				continue
			}
			// The check for embedded pointer types must
			// be deferred because of things like
			//  type T *struct { T }
			lateCheck := a.lateCheck
			a.lateCheck = func() bool {
				if _, ok := nt.lit().(*PtrType); ok {
					a.diagAt(&poss[i], "embedded type %v is a pointer type", nt)
					return false
				}
				return lateCheck()
			}
		}

		// Check name uniqueness
		if prev, ok := nameSet[name]; ok {
			a.diagAt(&poss[i], "field %s redeclared\n\tprevious declaration at %s", name, &prev)
			bad = true
			continue
		}
		nameSet[name] = poss[i]

		// Create field
		fields[i].Name = name
		fields[i].Type = ts[i]
		fields[i].Anonymous = (names[i] == nil)
	}

	if bad {
		return nil
	}

	return NewStructType(fields)
}

func (a *typeCompiler) compilePtrType(x *ast.StarExpr) Type {
	elem := a.compileType(x.X, true)
	if elem == nil {
		return nil
	}
	return NewPtrType(elem)
}

func (a *typeCompiler) compileFuncType(x *ast.FuncType, allowRec bool) *FuncDecl {
	// TODO(austin) Variadic function types

	// The types of parameters and results must be complete.
	//
	// TODO(austin) It's not clear they actually have to be complete.
	in, inNames, _, inBad := a.compileFields(x.Params, allowRec)
	out, outNames, _, outBad := a.compileFields(x.Results, allowRec)

	if inBad || outBad {
		return nil
	}
	return &FuncDecl{NewFuncType(in, false, out), nil, inNames, outNames}
}

func (a *typeCompiler) compileInterfaceType(x *ast.InterfaceType, allowRec bool) *InterfaceType {
	ts, names, poss, bad := a.compileFields(x.Methods, allowRec)

	methods := make([]IMethod, len(ts))
	nameSet := make(map[string]token.Position, len(ts))
	embeds := make([]*InterfaceType, len(ts))

	var nm, ne int
	for i := range ts {
		if ts[i] == nil {
			continue
		}

		if names[i] != nil {
			name := names[i].Value
			methods[nm].Name = name
			methods[nm].Type = ts[i].(*FuncType)
			nm++
			if prev, ok := nameSet[name]; ok {
				a.diagAt(&poss[i], "method %s redeclared\n\tprevious declaration at %s", name, &prev)
				bad = true
				continue
			}
			nameSet[name] = poss[i]
		} else {
			// Embedded interface
			it, ok := ts[i].lit().(*InterfaceType)
			if !ok {
				a.diagAt(&poss[i], "embedded type must be an interface")
				bad = true
				continue
			}
			embeds[ne] = it
			ne++
			for _, m := range it.methods {
				if prev, ok := nameSet[m.Name]; ok {
					a.diagAt(&poss[i], "method %s redeclared\n\tprevious declaration at %s", m.Name, &prev)
					bad = true
					continue
				}
				nameSet[m.Name] = poss[i]
			}
		}
	}

	if bad {
		return nil
	}

	methods = methods[0:nm]
	embeds = embeds[0:ne]

	return NewInterfaceType(methods, embeds)
}

func (a *typeCompiler) compileMapType(x *ast.MapType) Type {
	key := a.compileType(x.Key, true)
	val := a.compileType(x.Value, true)
	if key == nil || val == nil {
		return nil
	}
	// XXX(Spec) The Map types section explicitly lists all types
	// that can be map keys except for function types.
	switch key.lit().(type) {
	case *StructType:
		a.diagAt(x, "map key cannot be a struct type")
		return nil
	case *ArrayType:
		a.diagAt(x, "map key cannot be an array type")
		return nil
	case *SliceType:
		a.diagAt(x, "map key cannot be a slice type")
		return nil
	}
	return NewMapType(key, val)
}

func (a *typeCompiler) compileType(x ast.Expr, allowRec bool) Type {
	switch x := x.(type) {
	case *ast.BadExpr:
		// Error already reported by parser
		a.silentErrors++
		return nil

	case *ast.Ident:
		return a.compileIdent(x, allowRec)

	case *ast.ArrayType:
		return a.compileArrayType(x, allowRec)

	case *ast.StructType:
		return a.compileStructType(x, allowRec)

	case *ast.StarExpr:
		return a.compilePtrType(x)

	case *ast.FuncType:
		fd := a.compileFuncType(x, allowRec)
		if fd == nil {
			return nil
		}
		return fd.Type

	case *ast.InterfaceType:
		return a.compileInterfaceType(x, allowRec)

	case *ast.MapType:
		return a.compileMapType(x)

	case *ast.ChanType:
		goto notimpl

	case *ast.ParenExpr:
		return a.compileType(x.X, allowRec)

	case *ast.Ellipsis:
		a.diagAt(x, "illegal use of ellipsis")
		return nil
	}
	a.diagAt(x, "expression used as type")
	return nil

notimpl:
	a.diagAt(x, "compileType: %T not implemented", x)
	return nil
}

/*
 * Type compiler interface
 */

func noLateCheck() bool { return true }

func (a *compiler) compileType(b *block, typ ast.Expr) Type {
	tc := &typeCompiler{a, b, noLateCheck}
	t := tc.compileType(typ, false)
	if !tc.lateCheck() {
		t = nil
	}
	return t
}

func (a *compiler) compileTypeDecl(b *block, decl *ast.GenDecl) bool {
	ok := true
	for _, spec := range decl.Specs {
		spec := spec.(*ast.TypeSpec)
		// Create incomplete type for this type
		nt := b.DefineType(spec.Name.Value, spec.Name.Pos(), nil)
		if nt != nil {
			nt.(*NamedType).incomplete = true
		}
		// Compile type
		tc := &typeCompiler{a, b, noLateCheck}
		t := tc.compileType(spec.Type, false)
		if t == nil {
			// Create a placeholder type
			ok = false
		}
		// Fill incomplete type
		if nt != nil {
			nt.(*NamedType).Complete(t)
		}
		// Perform late type checking with complete type
		if !tc.lateCheck() {
			ok = false
			if nt != nil {
				// Make the type a placeholder
				nt.(*NamedType).Def = nil
			}
		}
	}
	return ok
}

func (a *compiler) compileFuncType(b *block, typ *ast.FuncType) *FuncDecl {
	tc := &typeCompiler{a, b, noLateCheck}
	res := tc.compileFuncType(typ, false)
	if res != nil {
		if !tc.lateCheck() {
			res = nil
		}
	}
	return res
}
