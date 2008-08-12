// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Importer

import Platform "platform"
import Utils "utils"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"


type Importer struct {
	comp *Globals.Compilation;
	debug bool;
	buf string;
	buf_pos int;
	pkg_list [256] *Globals.Package;
	pkg_ref int;
	type_list [1024] *Globals.Type;
	type_ref int;
};


func (I *Importer) ReadObject() *Globals.Object;


func (I *Importer) ReadByte() byte {
	x := I.buf[I.buf_pos];
	I.buf_pos++;
	/*
	if E.debug {
		print(" ", x);
	}
	*/
	return x;
}


func (I *Importer) ReadInt() int {
	x := 0;
	s := 0;  // TODO eventually Go will require this to be a uint!
	b := I.ReadByte();
	for b < 128 {
		x |= int(b) << s;
		s += 7;
		b = I.ReadByte();
	}
	// b >= 128
	x |= ((int(b) - 192) << s);
	/*
	if I.debug {
		print(" #", x);
	}
	*/
	return x;
}


func (I *Importer) ReadString() string {
	var buf [256] byte;  // TODO this needs to be fixed
	n := I.ReadInt();
	for i := 0; i < n; i++ {
		buf[i] = I.ReadByte();
	}
	s := string(buf)[0 : n];
	if I.debug {
		print(` "`, s, `"`);
	}
	return s;
}


func (I *Importer) ReadPackageTag() int {
	tag := I.ReadInt();
	if I.debug {
		if tag >= 0 {
			print(" [P", tag, "]");  // package ref
		} else {
			print("\nP", I.pkg_ref, ":");
		}
	}
	return tag;
}


func (I *Importer) ReadTypeTag() int {
	tag := I.ReadInt();
	if I.debug {
		if tag >= 0 {
			print(" [T", tag, "]");  // type ref
		} else {
			print("\nT", I.type_ref, ": ", Type.FormStr(-tag));
		}
	}
	return tag;
}


func (I *Importer) ReadObjectTag() int {
	tag := I.ReadInt();
	if tag < 0 {
		panic("tag < 0");
	}
	if I.debug {
		print("\n", Object.KindStr(tag));
	}
	return tag;
}


func (I *Importer) ReadPackage() *Globals.Package {
	tag := I.ReadPackageTag();
	if tag >= 0 {
		return I.pkg_list[tag];  // package already imported
	}

	ident := I.ReadString();
	file_name := I.ReadString();
	key := I.ReadString();
	
	// Canonicalize package - if it was imported before,
	// use the primary import.
	pkg := I.comp.Lookup(file_name);
	if pkg == nil {
		// new package
		obj := Globals.NewObject(-1, Object.PACKAGE, ident);
		pkg = Globals.NewPackage(file_name, obj, Globals.NewScope(nil));
		I.comp.Insert(pkg);
		if I.comp.flags.verbosity > 1 {
			print(`import: implicitly adding package `, ident, ` "`, file_name, `" (pno = `, obj.pnolev, ")\n");
		}
	} else if key != "" && key != pkg.key {
		// the package was imported before but the package
		// key has changed (a "" key indicates a forward-
		// declared package - it's key is consistent with
		// any actual package of the same name)
		panic("package key inconsistency");
	}
	I.pkg_list[I.pkg_ref] = pkg;
	I.pkg_ref++;

	return pkg;
}


func (I *Importer) ReadScope(scope *Globals.Scope, allow_multiples bool) {
	if I.debug {
		print(" {");
	}

	obj := I.ReadObject();
	for obj != nil {
		// allow_multiples is for debugging only - we should never
		// have multiple imports where we don't expect them
		if allow_multiples {
			scope.InsertImport(obj);
		} else {
			scope.Insert(obj);
		}
		obj = I.ReadObject();
	}
	
	if I.debug {
		print(" }");
	}
}


func (I *Importer) ReadType() *Globals.Type {
	tag := I.ReadTypeTag();
	if tag >= 0 {
		return I.type_list[tag];  // type already imported
	}

	typ := Globals.NewType(-tag);
	ptyp := typ;  // primary type

	ident := I.ReadString();
	if len(ident) > 0 {
		// named type
		pkg := I.ReadPackage();
		
		// create corresponding type object
		obj := Globals.NewObject(0, Object.TYPE, ident);
		obj.exported = true;
		obj.typ = typ;
		obj.pnolev = pkg.obj.pnolev;
		typ.obj = obj;

		// canonicalize type
		// (if the type was seen before, use primary instance!)
		ptyp = pkg.scope.InsertImport(obj).typ;
	}
	// insert the primary type into the type table but
	// keep filling in the current type fields
	I.type_list[I.type_ref] = ptyp;
	I.type_ref++;

	switch (typ.form) {
	case Type.FORWARD:
		typ.scope = Globals.NewScope(nil);
		break;
		
	case Type.ALIAS, Type.MAP:
		typ.aux = I.ReadType();
		typ.elt = I.ReadType();

	case Type.ARRAY:
		typ.len_ = I.ReadInt();
		typ.elt = I.ReadType();

	case Type.CHANNEL:
		typ.flags = I.ReadInt();
		typ.elt = I.ReadType();

	case Type.FUNCTION:
		typ.flags = I.ReadInt();
		typ.scope = Globals.NewScope(nil);
		I.ReadScope(typ.scope, false);

	case Type.STRUCT, Type.INTERFACE:
		typ.scope = Globals.NewScope(nil);
		I.ReadScope(typ.scope, false);

	case Type.POINTER, Type.REFERENCE:
		typ.elt = I.ReadType();

	default:
		panic("UNREACHABLE");
	}

	return ptyp;  // only use primary type
}


func (I *Importer) ReadObject() *Globals.Object {
	tag := I.ReadObjectTag();
	if tag == Object.END {
		return nil;
	}
	
	if tag == Object.TYPE {
		// named types are handled entirely by ReadType()
		typ := I.ReadType();
		if typ.obj.typ != typ {
			panic("inconsistent named type");
		}
		return typ.obj;
	}
	
	ident := I.ReadString();
	obj := Globals.NewObject(0, tag, ident);
	obj.exported = true;
	obj.typ = I.ReadType();

	switch (tag) {
	case Object.CONST:
		I.ReadInt();  // should set the value field

	case Object.VAR, Object.FIELD:
		I.ReadInt();  // should set the address/offset field

	case Object.FUNC:
		I.ReadInt();  // should set the address/offset field
		
	default:
		panic("UNREACHABLE");
	}

	return obj;
}


func (I *Importer) Import(comp* Globals.Compilation, data string) *Globals.Package {
	I.comp = comp;
	I.debug = comp.flags.debug;
	I.buf = data;
	I.buf_pos = 0;
	I.pkg_ref = 0;
	I.type_ref = 0;
	
	// check magic bits
	if !Utils.Contains(data, Platform.MAGIC_obj_file, 0) {
		return nil;
	}
	
	// Predeclared types are "pre-imported".
	for p := Universe.types.first; p != nil; p = p.next {
		if p.typ.ref != I.type_ref {
			panic("incorrect ref for predeclared type");
		}
		I.type_list[I.type_ref] = p.typ;
		I.type_ref++;
	}

	// import package
	pkg := I.ReadPackage();
	I.ReadScope(pkg.scope, true);
	
	if I.debug {
		print("\n(", I.buf_pos, " bytes)\n");
	}
	
	return pkg;
}


export func Import(comp *Globals.Compilation, data string) *Globals.Package {
	var I Importer;
	pkg := (&I).Import(comp, data);
	return pkg;
}
