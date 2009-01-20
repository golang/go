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
	panic("internal compiler error: ", msg, "\n");
}


type Verifier struct {
	comp *Globals.Compilation;

	// various sets for marking the graph (and thus avoid cycles)
	objs map[*Globals.Object] bool;
	typs map[*Globals.Type] bool;
	pkgs map[*Globals.Package] bool;
}


func (V *Verifier) VerifyObject(obj *Globals.Object, pnolev int);


func (V *Verifier) VerifyType(typ *Globals.Type) {
	if present, ok := V.typs[typ]; present {
		return;  // already verified
	}
	V.typs[typ] = true;

	if typ.obj != nil {
		V.VerifyObject(typ.obj, 0);
	}

	switch typ.form {
	case Type.VOID:
	case Type.BAD:
		break;  // TODO for now - remove eventually

	case Type.FORWARD:
		if typ.scope == nil {
			Error("forward types must have a scope");
		}

	case Type.TUPLE:
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
	default:
		Error("illegal type form " + Type.FormStr(typ.form));
	}
}


func (V *Verifier) VerifyObject(obj *Globals.Object, pnolev int) {
	if present, ok := V.objs[obj]; present {
		return;  // already verified
	}
	V.objs[obj] = true;

	// all objects have a non-nil type
	V.VerifyType(obj.typ);

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


func (V *Verifier) VerifyScope(scope *Globals.Scope) {
	for p := scope.entries.first; p != nil; p = p.next {
		V.VerifyObject(p.obj, 0);
	}
}


func (V *Verifier) VerifyPackage(pkg *Globals.Package, pno int) {
	if present, ok := V.pkgs[pkg]; present {
		return;  // already verified
	}
	V.pkgs[pkg] = true;

	V.VerifyObject(pkg.obj, pno);
	V.VerifyScope(pkg.scope);
}


func (V *Verifier) Verify(comp *Globals.Compilation) {
	// initialize Verifier
	V.comp = comp;
	V.objs = make(map[*Globals.Object] bool);
	V.typs = make(map[*Globals.Type] bool);
	V.pkgs = make(map[*Globals.Package] bool);

	// verify all packages
	filenames := make(map[string] bool);
	for i := 0; i < comp.pkg_ref; i++ {
		pkg := comp.pkg_list[i];
		// each pkg filename must appear only once
		if present, ok := filenames[pkg.file_name]; present {
			Error("package filename present more than once");
		}
		filenames[pkg.file_name] = true;
		V.VerifyPackage(pkg, i);
	}
}


func Verify(comp *Globals.Compilation) {
	V := new(Verifier);
	V.Verify(comp);
}
