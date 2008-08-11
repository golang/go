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


type Verifier struct {
	comp *Globals.Compilation;
	
	// various sets for marking the graph (and thus avoid cycles)
	objs *map[*Globals.Object] bool;
	typs *map[*Globals.Type] bool;
	pkgs *map[*Globals.Package] bool;
}


func (V *Verifier) VerifyObject(obj *Globals.Object, pnolev int);


func (V *Verifier) VerifyType(typ *Globals.Type) {
	if V.typs[typ] {
		return;  // already verified
	}
	V.typs[typ] = true;
	
	if typ.obj != nil {
		V.VerifyObject(typ.obj, 0);
	}
	
	switch typ.form {
	case Type.VOID:
		break;  // TODO for now - remove eventually
	case Type.FORWARD:
		if typ.scope == nil {
			Error("forward types must have a scope");
		}
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


func (V *Verifier) VerifyObject(obj *Globals.Object, pnolev int) {
	if V.objs[obj] {
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
	if V.pkgs[pkg] {
		return;  // already verified
	}
	V.pkgs[pkg] = true;
	
	V.VerifyObject(pkg.obj, pno);
	V.VerifyScope(pkg.scope);
}


func (V *Verifier) Verify(comp *Globals.Compilation) {
	// initialize Verifier
	V.comp = comp;
	V.objs = new(map[*Globals.Object] bool);
	V.typs = new(map[*Globals.Type] bool);
	V.pkgs = new(map[*Globals.Package] bool);

	// verify all packages
	filenames := new(map[string] bool);
	for i := 0; i < comp.pkg_ref; i++ {
		pkg := comp.pkg_list[i];
		// each pkg filename must appear only once
		if filenames[pkg.file_name] {
			Error("package filename present more then once");
		}
		filenames[pkg.file_name] = true;
		V.VerifyPackage(pkg, i);
	}
}


export func Verify(comp *Globals.Compilation) {
	V := new(Verifier);
	V.Verify(comp);
}
