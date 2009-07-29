// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
	"go/ast";
	"log";
)


/*
 * Type compiler
 */

// TODO(austin) Without this, I get a "conflicting definitions for
// eval.compiler" when gopack'ing typec.6 from gobuild.
// Interestingly, if I create the Makefile with this line, then
// comment it out and use the Makefile, things work.
type exprCompiler struct

type typeCompiler struct {
	*compiler;
	block *block;
}

func (a *typeCompiler) compileType(x ast.Expr) Type

func (a *typeCompiler) compileIdent(x *ast.Ident) Type {
	_, def := a.block.Lookup(x.Value);
	if def == nil {
		a.diagAt(x, "%s: undefined", x.Value);
		return nil;
	}
	switch def := def.(type) {
	case *Constant:
		a.diagAt(x, "constant %v used as type", x.Value);
		return nil;
	case *Variable:
		a.diagAt(x, "variable %v used as type", x.Value);
		return nil;
	case Type:
		return def;
	}
	log.Crashf("name %s has unknown type %T", x.Value, def);
	return nil;
}

func (a *typeCompiler) compileArrayType(x *ast.ArrayType) *ArrayType {
	// Compile length expression
	if x.Len == nil {
		a.diagAt(x, "slice types not implemented");
		return nil;
	}
	if _, ok := x.Len.(*ast.Ellipsis); ok {
		a.diagAt(x.Len, "... array initailizers not implemented");
		return nil;
	}
	l, ok := a.compileArrayLen(a.block, x.Len);

	// Compile element type
	elem := a.compileType(x.Elt);

	if !ok {
		return nil;
	}
	if l < 0 {
		a.diagAt(x.Len, "array length must be non-negative");
		return nil;
	}
	if elem == nil {
		return nil;
	}

	return NewArrayType(l, elem);
}

func (a *typeCompiler) compilePtrType(x *ast.StarExpr) *PtrType {
	elem := a.compileType(x.X);
	if elem == nil {
		return nil;
	}
	return NewPtrType(elem);
}

func countFields(fs []*ast.Field) int {
	n := 0;
	for _, f := range fs {
		if f.Names == nil {
			n++;
		} else {
			n += len(f.Names);
		}
	}
	return n;
}

func (a *typeCompiler) compileFields(fs []*ast.Field) ([]Type, []*ast.Ident) {
	n := countFields(fs);
	ts := make([]Type, n);
	ns := make([]*ast.Ident, n);

	bad := false;
	i := 0;
	for fi, f := range fs {
		t := a.compileType(f.Type);
		if t == nil {
			bad = true;
		}
		if f.Names == nil {
			// TODO(austin) In a struct, this has an
			// implicit name.  However, this also triggers
			// for function return values, which should
			// not be given names.
			ns[i] = nil;
			ts[i] = t;
			i++;
			continue;
		}
		for _, n := range f.Names {
			ns[i] = n;
			ts[i] = t;
			i++;
		}
	}

	if bad {
		return nil, nil;
	}
	return ts, ns;
}

func (a *typeCompiler) compileFuncType(x *ast.FuncType) *FuncDecl {
	// TODO(austin) Variadic function types

	bad := false;

	in, inNames := a.compileFields(x.Params);
	out, outNames := a.compileFields(x.Results);

	if in == nil || out == nil {
		return nil;
	}
	return &FuncDecl{NewFuncType(in, false, out), nil, inNames, outNames};
}

func (a *typeCompiler) compileType(x ast.Expr) Type {
	switch x := x.(type) {
	case *ast.Ident:
		return a.compileIdent(x);

	case *ast.ArrayType:
		return a.compileArrayType(x);

	case *ast.StructType:
		goto notimpl;

	case *ast.StarExpr:
		return a.compilePtrType(x);

	case *ast.FuncType:
		return a.compileFuncType(x).Type;

	case *ast.InterfaceType:
		goto notimpl;

	case *ast.MapType:
		goto notimpl;

	case *ast.ChanType:
		goto notimpl;

	case *ast.ParenExpr:
		return a.compileType(x.X);

	case *ast.Ellipsis:
		a.diagAt(x, "illegal use of ellipsis");
		return nil;
	}
	a.diagAt(x, "expression used as type");
	return nil;

notimpl:
	a.diagAt(x, "compileType: %T not implemented", x);
	return nil;
}

/*
 * Type compiler interface
 */

func (a *compiler) compileType(b *block, typ ast.Expr) Type {
	tc := &typeCompiler{a, b};
	return tc.compileType(typ);
}

func (a *compiler) compileFuncType(b *block, typ *ast.FuncType) *FuncDecl {
	tc := &typeCompiler{a, b};
	return tc.compileFuncType(typ);
}
