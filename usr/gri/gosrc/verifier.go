// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verifies compiler-internal data structures.

package Verifier

import Utils "utils"
import Scanner "scanner"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"
import Import "import"
import AST "ast"


func Error(msg string) {
	panic "internal compiler error: ", msg, "\n";
}


func VerifyObject(obj *Globals.Object, pnolev int);


func VerifyType(typ *Globals.Type) {
	if typ.obj != nil {
		VerifyObject(typ.obj, 0);
	}
	
	switch typ.form {
	case Type.UNDEF:  // for now - remove eventually
		break;
	case Type.NIL:
		break;
	case Type.BOOL:
		break;
	case Type.UINT:
		break;
	case Type.INT:
		break;
	case Type.FLOAT:
		break;
	case Type.STRING:
		break;
	case Type.ANY:
		break;
	case Type.ALIAS:
		break;
	case Type.ARRAY:
		break;
	case Type.STRUCT:
		break;
	case Type.INTERFACE:
		break;
	case Type.MAP:
		break;
	case Type.CHANNEL:
		break;
	case Type.FUNCTION:
		break;
	case Type.POINTER:
		break;
	case Type.REFERENCE:
		break;
	default:
		Error("illegal type form " + Type.FormStr(typ.form));
	}
}


func VerifyObject(obj *Globals.Object, pnolev int) {
	VerifyType(obj.typ);
	
	switch obj.kind {
	case Object.CONST:
		break;
	case Object.TYPE:
		break;
	case Object.VAR:
		break;
	case Object.FUNC:
		break;
	case Object.PACKAGE:
		break;
	case Object.LABEL:
		break;
	default:
		Error("illegal object kind " + Object.KindStr(obj.kind));
	}
}


func VerifyScope(scope *Globals.Scope) {
	for p := scope.entries.first; p != nil; p = p.next {
		VerifyObject(p.obj, 0);
	}
}


func VerifyPackage(pkg *Globals.Package, pno int) {
	VerifyObject(pkg.obj, 0);
}


export Verify
func Verify(comp *Globals.Compilation) {
	for i := 0; i < comp.pkg_ref; i++ {
		VerifyPackage(comp.pkg_list[i], i);
	}
}
