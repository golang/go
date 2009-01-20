// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Expr

import Globals "globals"
import Universe "universe"
import Object "object"
import Type "type"
import AST "ast"


// TODO the following shortcuts below don't work due to 6g/6l bugs
//type Compilation Globals.Compilation
//type Expr Globals.Expr


func Error(comp *Globals.Compilation, pos int, msg string) {
	comp.env.Error(comp, pos, msg);
}


func Deref(comp *Globals.Compilation, x Globals.Expr) Globals.Expr {
	switch typ := x.typ(); typ.form {
	case Type.BAD:
		// ignore

	case Type.POINTER:
		x = AST.NewDeref(x);

	default:
		Error(comp, x.pos(), `"*" not applicable (typ.form = ` + Type.FormStr(typ.form) + `)`);
		x = AST.Bad;
	}

	return x;
}


func Select(comp *Globals.Compilation, x Globals.Expr, pos int, selector string) Globals.Expr {
	if x.typ().form == Type.POINTER {
		x = Deref(comp, x);
	}

	switch typ := x.typ(); typ.form {
	case Type.BAD:
		// ignore

	case Type.STRUCT, Type.INTERFACE:
		obj := typ.scope.Lookup(selector);
		if obj != nil {
			x = AST.NewSelector(x.pos(), obj.typ);

		} else {
			Error(comp, pos, `no field/method "` + selector + `"`);
			x = AST.Bad;
		}

	default:
		Error(comp, pos, `"." not applicable (typ.form = ` + Type.FormStr(typ.form) + `)`);
		x = AST.Bad;
	}

	return x;
}


func AssertType(comp *Globals.Compilation, x Globals.Expr, pos int, typ *Globals.Type) Globals.Expr {
	return AST.Bad;
}


func Index(comp *Globals.Compilation, x, i Globals.Expr) Globals.Expr {
	if x.typ().form == Type.POINTER {
		x = Deref(comp, x);
	}

	switch typ := x.typ(); typ.form {
	case Type.BAD:
		// ignore

	case Type.STRING, Type.ARRAY:
		x = AST.Bad;

	case Type.MAP:
		if Type.Equal(typ.key, i.typ()) {
			// x = AST.NewSubscript(x, i1);
			x = AST.Bad;

		} else {
			Error(comp, x.pos(), "map key type mismatch");
			x = AST.Bad;
		}

	default:
		Error(comp, x.pos(), `"[]" not applicable (typ.form = ` + Type.FormStr(typ.form) + `)`);
		x = AST.Bad;
	}
	return x;
}


func Slice(comp *Globals.Compilation, x, i, j Globals.Expr) Globals.Expr {
	if x.typ().form == Type.POINTER {
		x = Deref(comp, x);
	}

	switch typ := x.typ(); typ.form {
	case Type.BAD:
		// ignore
		break;
	case Type.STRING, Type.ARRAY:
		x = AST.Bad;

	case Type.MAP:
		if Type.Equal(typ.key, i.typ()) {
			// x = AST.NewSubscript(x, i1);
			x = AST.Bad;

		} else {
			Error(comp, x.pos(), "map key type mismatch");
			x = AST.Bad;
		}

	default:
		Error(comp, x.pos(), `"[:]" not applicable (typ.form = ` + Type.FormStr(typ.form) + `)`);
		x = AST.Bad;
	}
	return x;
}


func Call(comp *Globals.Compilation, x Globals.Expr, args *Globals.List) Globals.Expr {
	if x.typ().form == Type.POINTER {
		x = Deref(comp, x);
	}

	if x.op() == AST.OBJECT && x.(*AST.Object).obj.kind == Object.BUILTIN {
		panic("builtin call - UNIMPLEMENTED");
	}

	typ := x.typ();
	if typ.form == Type.FUNCTION || typ.form == Type.METHOD {
		// TODO check args against parameters
	}

	return AST.Bad;
}


func UnaryExpr(comp *Globals.Compilation, x Globals.Expr) Globals.Expr {
	return AST.Bad;
}


func BinaryExpr(comp *Globals.Compilation, x, y Globals.Expr) Globals.Expr {
	e := new(AST.BinaryExpr);
	e.typ_ = x.typ();  // TODO fix this
	//e.op = P.tok;  // TODO should we use tokens or separate operator constants?
	e.x = x;
	e.y = y;
	return e;
}
