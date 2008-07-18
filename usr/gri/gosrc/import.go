// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Importer

import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"


type Importer struct {
	comp *Globals.Compilation;
	debug bool;
	buf string;
	pos int;
	pkgs [256] *Globals.Package;
	npkgs int;
	types [1024] *Globals.Type;
	ntypes int;
};


func (I *Importer) ReadType() *Globals.Type;
func (I *Importer) ReadObject(tag int) *Globals.Object;
func (I *Importer) ReadPackage() *Globals.Package;


func (I *Importer) ReadByte() byte {
	x := I.buf[I.pos];
	I.pos++;
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


func (I *Importer) ReadObjTag() int {
	tag := I.ReadInt();
	if tag < 0 {
		panic "tag < 0";
	}
	if I.debug {
		print "\nObj: ", tag;  // obj kind
	}
	return tag;
}


func (I *Importer) ReadTypeTag() int {
	tag := I.ReadInt();
	if I.debug {
		if tag > 0 {
			print "\nTyp ", I.ntypes, ": ", tag;  // type form
		} else {
			print " [Typ ", -tag, "]";  // type ref
		}
	}
	return tag;
}


func (I *Importer) ReadPackageTag() int {
	tag := I.ReadInt();
	if I.debug {
		if tag > 0 {
			print "\nPkg ", I.npkgs, ": ", tag;  // package tag
		} else {
			print " [Pkg ", -tag, "]";  // package ref
		}
	}
	return tag;
}


func (I *Importer) ReadTypeField() *Globals.Object {
	fld := Globals.NewObject(0, Object.VAR, "");
	fld.typ = I.ReadType();
	return fld;
}


func (I *Importer) ReadScope() *Globals.Scope {
	if I.debug {
		print " {";
	}

	scope := Globals.NewScope(nil);
	for n := I.ReadInt(); n > 0; n-- {
		tag := I.ReadObjTag();
		scope.Insert(I.ReadObject(tag));
	}

	if I.debug {
		print " }";
	}
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
		default: fallthrough;
		case Object.BAD: fallthrough;
		case Object.PACKAGE: fallthrough;
		case Object.PTYPE:
			panic "UNREACHABLE";

		case Object.CONST:
			I.ReadInt();  // should set the value field

		case Object.TYPE:
			// nothing to do
			
		case Object.VAR:
			I.ReadInt();  // should set the address/offset field

		case Object.FUNC:
			I.ReadInt();  // should set the address/offset field
		}

		return obj;
	}
}


func (I *Importer) ReadType() *Globals.Type {
	tag := I.ReadTypeTag();

	if tag <= 0 {
		return I.types[-tag];  // type already imported
	}

	typ := Globals.NewType(tag);
	ptyp := typ;  // primary type
	ident := I.ReadString();
	if (len(ident) > 0) {
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
	I.types[I.ntypes] = ptyp;
	I.ntypes++;

	switch (tag) {
	default: fallthrough;
	case Type.UNDEF: fallthrough;
	case Type.BAD: fallthrough;
	case Type.NIL: fallthrough;
	case Type.BOOL: fallthrough;
	case Type.UINT: fallthrough;
	case Type.INT: fallthrough;
	case Type.FLOAT: fallthrough;
	case Type.STRING: fallthrough;
	case Type.ANY:
		panic "UNREACHABLE";

	case Type.ARRAY:
		typ.len_ = I.ReadInt();
		typ.elt = I.ReadTypeField();

	case Type.MAP:
		typ.key = I.ReadTypeField();
		typ.elt = I.ReadTypeField();

	case Type.CHANNEL:
		typ.flags = I.ReadInt();
		typ.elt = I.ReadTypeField();

	case Type.FUNCTION:
		typ.flags = I.ReadInt();
		fallthrough;
	case Type.STRUCT: fallthrough;
	case Type.INTERFACE:
		typ.scope = I.ReadScope();

	case Type.POINTER: fallthrough;
	case Type.REFERENCE:
		typ.elt = I.ReadTypeField();
	}

	return ptyp;  // only use primary type
}


func (I *Importer) ReadPackage() *Globals.Package {
	tag := I.ReadPackageTag();

	if (tag <= 0) {
		return I.pkgs[-tag];  // package already imported
	}

	ident := I.ReadString();
	file_name := I.ReadString();
	key := I.ReadString();
	pkg := I.comp.Lookup(file_name);

	if pkg == nil {
		// new package
		pkg = Globals.NewPackage(file_name);
		pkg.scope = Globals.NewScope(nil);
		pkg = I.comp.InsertImport(pkg);

	} else if (key != pkg.key) {
		// package inconsistency
		panic "package key inconsistency";
	}
	I.pkgs[I.npkgs] = pkg;
	I.npkgs++;

	return pkg;
}


func (I *Importer) Import(comp* Globals.Compilation, file_name string) {
	if I.debug {
		print "importing from ", file_name;
	}
	
	buf, ok := sys.readfile(file_name);
	if !ok {
		panic "import failed";
	}
	
	I.comp = comp;
	I.debug = true;
	I.buf = buf;
	I.pos = 0;
	I.npkgs = 0;
	I.ntypes = 0;
	
	// Predeclared types are "pre-exported".
	for p := Universe.types.first; p != nil; p = p.next {
		if p.typ.ref != I.ntypes {
			panic "incorrect ref for predeclared type";
		}
		I.types[I.ntypes] = p.typ;
		I.ntypes++;
	}

	pkg := I.ReadPackage();
	for {
		tag := I.ReadObjTag();
		if tag == 0 {
			break;
		}
		obj := I.ReadObject(tag);
		obj.pnolev = pkg.obj.pnolev;
		pkg.scope.InsertImport(obj);
	}

	if I.debug {
		print "\n(", I.pos, " bytes)\n";
	}
}
