// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/internal/typeparams"
	"go/token"
)

func (check *Checker) interfaceType(ityp *Interface, iface *ast.InterfaceType, def *Named) {
	var tlist []ast.Expr
	var tname *ast.Ident // "type" name of first entry in a type list declaration

	addEmbedded := func(pos token.Pos, typ Type) {
		ityp.embeddeds = append(ityp.embeddeds, typ)
		if ityp.embedPos == nil {
			ityp.embedPos = new([]token.Pos)
		}
		*ityp.embedPos = append(*ityp.embedPos, pos)
	}

	for _, f := range iface.Methods.List {
		if len(f.Names) == 0 {
			// We have an embedded type; possibly a union of types.
			addEmbedded(f.Type.Pos(), parseUnion(check, flattenUnion(nil, f.Type)))
			continue
		}

		// We have a method with name f.Names[0], or a type
		// of a type list (name.Name == "type").
		// (The parser ensures that there's only one method
		// and we don't care if a constructed AST has more.)
		name := f.Names[0]
		if name.Name == "_" {
			check.errorf(name, _BlankIfaceMethod, "invalid method name _")
			continue // ignore
		}

		// TODO(rfindley) Remove type list handling once the parser doesn't accept type lists anymore.
		if name.Name == "type" {
			// Report an error for the first type list per interface
			// if we don't allow type lists, but continue.
			if !allowTypeLists && tlist == nil {
				check.softErrorf(name, _Todo, "use generalized embedding syntax instead of a type list")
			}
			// For now, collect all type list entries as if it
			// were a single union, where each union element is
			// of the form ~T.
			// TODO(rfindley) remove once we disallow type lists
			op := new(ast.UnaryExpr)
			op.Op = token.TILDE
			op.X = f.Type
			tlist = append(tlist, op)
			// Report an error if we have multiple type lists in an
			// interface, but only if they are permitted in the first place.
			if allowTypeLists && tname != nil && tname != name {
				check.errorf(name, _Todo, "cannot have multiple type lists in an interface")
			}
			tname = name
			continue
		}

		typ := check.typ(f.Type)
		sig, _ := typ.(*Signature)
		if sig == nil {
			if typ != Typ[Invalid] {
				check.invalidAST(f.Type, "%s is not a method signature", typ)
			}
			continue // ignore
		}

		// Always type-check method type parameters but complain if they are not enabled.
		// (This extra check is needed here because interface method signatures don't have
		// a receiver specification.)
		if sig.tparams != nil {
			var at positioner = f.Type
			if tparams := typeparams.Get(f.Type); tparams != nil {
				at = tparams
			}
			check.errorf(at, _Todo, "methods cannot have type parameters")
		}

		// use named receiver type if available (for better error messages)
		var recvTyp Type = ityp
		if def != nil {
			recvTyp = def
		}
		sig.recv = NewVar(name.Pos(), check.pkg, "", recvTyp)

		m := NewFunc(name.Pos(), check.pkg, name.Name, sig)
		check.recordDef(name, m)
		ityp.methods = append(ityp.methods, m)
	}

	// type constraints
	if tlist != nil {
		// TODO(rfindley): this differs from types2 due to the use of Pos() below,
		// which should actually be on the ~. Confirm that this position is correct.
		addEmbedded(tlist[0].Pos(), parseUnion(check, tlist))
	}

	// All methods and embedded elements for this interface are collected;
	// i.e., this interface is may be used in a type set computation.
	ityp.complete = true

	if len(ityp.methods) == 0 && len(ityp.embeddeds) == 0 {
		// empty interface
		ityp.tset = &topTypeSet
		return
	}

	// sort for API stability
	sortMethods(ityp.methods)
	// (don't sort embeddeds: they must correspond to *embedPos entries)

	// Compute type set with a non-nil *Checker as soon as possible
	// to report any errors. Subsequent uses of type sets should be
	// using this computed type set and won't need to pass in a *Checker.
	check.later(func() { newTypeSet(check, iface.Pos(), ityp) })
}

func flattenUnion(list []ast.Expr, x ast.Expr) []ast.Expr {
	if o, _ := x.(*ast.BinaryExpr); o != nil && o.Op == token.OR {
		list = flattenUnion(list, o.X)
		x = o.Y
	}
	return append(list, x)
}
