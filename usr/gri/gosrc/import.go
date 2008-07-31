// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Importer

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
	pkgs [256] *Globals.Package;
	pkg_ref int;
	types [1024] *Globals.Type;
	type_ref int;
};


func (I *Importer) ReadType() *Globals.Type;
func (I *Importer) ReadObject(tag int) *Globals.Object;
func (I *Importer) ReadPackage() *Globals.Package;


func (I *Importer) ReadByte() byte {
	x := I.buf[I.buf_pos];
	I.buf_pos++;
	/*
	if E.debug {
		print " ", x;
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
		print " #", x;
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
		print ` "`, s, `"`;
	}
	return s;
}


func (I *Importer) ReadObjectTag() int {
	tag := I.ReadInt();
	if tag < 0 {
		panic "tag < 0";
	}
	if I.debug {
		print "\n", Object.KindStr(tag);
	}
	return tag;
}


func (I *Importer) ReadTypeTag() int {
	tag := I.ReadInt();
	if I.debug {
		if tag >= 0 {
			print " [T", tag, "]";  // type ref
		} else {
			print "\nT", I.type_ref, ": ", Type.FormStr(-tag);
		}
	}
	return tag;
}


func (I *Importer) ReadPackageTag() int {
	tag := I.ReadInt();
	if I.debug {
		if tag >= 0 {
			print " [P", tag, "]";  // package ref
		} else {
			print "\nP", I.pkg_ref, ": ", -tag;  // package tag
		}
	}
	return tag;
}


func (I *Importer) ReadScope() *Globals.Scope {
	if I.debug {
		print " {";
	}

	scope := Globals.NewScope(nil);
	for {
		tag := I.ReadObjectTag();
		if tag == Object.EOS {  // terminator
			break;
		}
		// InsertImport only needed for package scopes
		// but ok to use always
		scope.InsertImport(I.ReadObject(tag));
	}
	
	if I.debug {
		print " }";
	}
	
	return scope;
}


func (I *Importer) ReadObject(tag int) *Globals.Object {
	if tag == Object.PTYPE {
		// primary type object - handled entirely by ReadType()
		typ := I.ReadType();
		if typ.obj.typ != typ {
			panic "incorrect primary type";
		}
		return typ.obj;

	} else {
		ident := I.ReadString();
		obj := Globals.NewObject(0, tag, ident);
		obj.typ = I.ReadType();
		obj.pnolev = I.ReadPackage().obj.pnolev;

		switch (tag) {
		case Object.CONST:
			I.ReadInt();  // should set the value field

		case Object.TYPE:
			// nothing to do
			
		case Object.VAR:
			I.ReadInt();  // should set the address/offset field

		case Object.FUNC:
			I.ReadInt();  // should set the address/offset field
			
		default:
			panic "UNREACHABLE";
		}

		return obj;
	}
}


func (I *Importer) ReadType() *Globals.Type {
	tag := I.ReadTypeTag();

	if tag >= 0 {
		return I.types[tag];  // type already imported
	}

	typ := Globals.NewType(-tag);
	ptyp := typ;  // primary type
	ident := I.ReadString();
	if len(ident) > 0 {
		// primary type
		obj := Globals.NewObject(0, Object.TYPE, ident);
		obj.typ = typ;
		typ.obj = obj;

		// canonicalize type
		pkg := I.ReadPackage();
		obj.pnolev = pkg.obj.pnolev;
		obj = pkg.scope.InsertImport(obj);

		ptyp = obj.typ;
	}
	I.types[I.type_ref] = ptyp;
	I.type_ref++;

	switch (typ.form) {
	default: fallthrough;
	case Type.ARRAY:
		typ.len_ = I.ReadInt();
		typ.elt = I.ReadType();

	case Type.MAP:
		typ.key = I.ReadType();
		typ.elt = I.ReadType();

	case Type.CHANNEL:
		typ.flags = I.ReadInt();
		typ.elt = I.ReadType();

	case Type.FUNCTION:
		typ.flags = I.ReadInt();
		typ.scope = I.ReadScope();

	case Type.STRUCT, Type.INTERFACE:
		typ.scope = I.ReadScope();

	case Type.POINTER, Type.REFERENCE:
		typ.elt = I.ReadType();

	default:
		panic "UNREACHABLE";
	}

	return ptyp;  // only use primary type
}


func (I *Importer) ReadPackage() *Globals.Package {
	tag := I.ReadPackageTag();

	if tag >= 0 {
		return I.pkgs[tag];  // package already imported
	}

	if -tag != Object.PACKAGE {
		panic "incorrect package tag";
	}
	
	ident := I.ReadString();
	file_name := I.ReadString();
	key := I.ReadString();
	pkg := I.comp.Lookup(file_name);

	if pkg == nil {
		// new package
		pkg = Globals.NewPackage(file_name);
		pkg.obj = Globals.NewObject(-1, Object.PACKAGE, ident);
		pkg.scope = Globals.NewScope(nil);
		pkg = I.comp.InsertImport(pkg);

	} else if key != pkg.key {
		// package inconsistency
		panic "package key inconsistency";
	}
	I.pkgs[I.pkg_ref] = pkg;
	I.pkg_ref++;

	return pkg;
}


func (I *Importer) Import(comp* Globals.Compilation, file_name string) *Globals.Package {
	I.comp = comp;
	I.debug = comp.flags.debug;
	I.buf = "";
	I.buf_pos = 0;
	I.pkg_ref = 0;
	I.type_ref = 0;
	
	if I.debug {
		print "importing from ", file_name, "\n";
	}
	
	buf, ok := sys.readfile(file_name);
	if !ok {
		return nil;
	}
	I.buf = buf;
	
	// Predeclared types are "pre-imported".
	for p := Universe.types.first; p != nil; p = p.next {
		if p.typ.ref != I.type_ref {
			panic "incorrect ref for predeclared type";
		}
		I.types[I.type_ref] = p.typ;
		I.type_ref++;
	}

	pkg := I.ReadPackage();
	for {
		tag := I.ReadObjectTag();
		if tag == Object.EOS {
			break;
		}
		obj := I.ReadObject(tag);
		obj.pnolev = pkg.obj.pnolev;
		pkg.scope.InsertImport(obj);
	}

	if I.debug {
		print "\n(", I.buf_pos, " bytes)\n";
	}
	
	return pkg;
}


export Import
func Import(comp* Globals.Compilation, pkg_name string) *Globals.Package {
	var I Importer;
	return (&I).Import(comp, Utils.FixExt(pkg_name));
}
