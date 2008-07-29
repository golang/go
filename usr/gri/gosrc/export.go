// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Exporter

import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"


export Exporter  // really only want to export Export()
type Exporter struct {
	comp *Globals.Compilation;
	debug bool;
	buf [4*1024] byte;
	pos int;
	pkg_ref int;
	type_ref int;
};


func (E *Exporter) WriteType(typ *Globals.Type);
func (E *Exporter) WriteObject(obj *Globals.Object);
func (E *Exporter) WritePackage(pkg *Globals.Package);


func (E *Exporter) WriteByte(x byte) {
	E.buf[E.pos] = x;
	E.pos++;
	/*
	if E.debug {
		print " ", x;
	}
	*/
}


func (E *Exporter) WriteInt(x int) {
	/*
	if E.debug {
		print " #", x;
	}
	*/
	for x < -64 || x >= 64 {
		E.WriteByte(byte(x & 127));
		x = int(uint(x >> 7));  // arithmetic shift
	}
	// -64 <= x && x < 64
	E.WriteByte(byte(x + 192));
}


func (E *Exporter) WriteString(s string) {
	if E.debug {
		print ` "`, s, `"`;
	}
	n := len(s);
	E.WriteInt(n);
	for i := 0; i < n; i++ {
		E.WriteByte(s[i]);
	}
}


func (E *Exporter) WriteObjTag(tag int) {
	if tag < 0 {
		panic "tag < 0";
	}
	if E.debug {
		print "\nObj: ", tag;  // obj kind
	}
	E.WriteInt(tag);
}


func (E *Exporter) WriteTypeTag(tag int) {
	if E.debug {
		if tag > 0 {
			print "\nTyp ", E.type_ref, ": ", tag;  // type form
		} else {
			print " [Typ ", -tag, "]";  // type ref
		}
	}
	E.WriteInt(tag);
}


func (E *Exporter) WritePackageTag(tag int) {
	if E.debug {
		if tag > 0 {
			print "\nPkg ", E.pkg_ref, ": ", tag;  // package no
		} else {
			print " [Pkg ", -tag, "]";  // package ref
		}
	}
	E.WriteInt(tag);
}


func (E *Exporter) WriteTypeField(fld *Globals.Object) {
	if fld.kind != Object.VAR {
		panic "fld.kind != Object.VAR";
	}
	E.WriteType(fld.typ);
}


func (E *Exporter) WriteScope(scope *Globals.Scope) {
	if E.debug {
		print " {";
	}

	// determine number of objects to export
	n := 0;
	for p := scope.entries.first; p != nil; p = p.next {
		if p.obj.exported {
			n++;
		}			
	}
	
	// export the objects, if any
	if n > 0 {
		for p := scope.entries.first; p != nil; p = p.next {
			if p.obj.exported {
				E.WriteObject(p.obj);
			}			
		}
	}

	if E.debug {
		print " }";
	}
}


func (E *Exporter) WriteObject(obj *Globals.Object) {
	if obj == nil || !obj.exported {
		panic "obj == nil || !obj.exported";
	}

	if obj.kind == Object.TYPE && obj.typ.obj == obj {
		// primary type object - handled entirely by WriteType()
		E.WriteObjTag(Object.PTYPE);
		E.WriteType(obj.typ);

	} else {
		E.WriteObjTag(obj.kind);
		E.WriteString(obj.ident);
		E.WriteType(obj.typ);
		//E.WritePackage(E.comp.pkgs[obj.pnolev]);

		switch obj.kind {
		case Object.CONST:
			E.WriteInt(0);  // should be the correct value

		case Object.TYPE:
			// nothing to do
			
		case Object.VAR:
			E.WriteInt(0);  // should be the correct address/offset
			
		case Object.FUNC:
			E.WriteInt(0);  // should be the correct address/offset
			
		default:
			print "obj.kind = ", obj.kind, "\n";
			panic "UNREACHABLE";
		}
	}
}


func (E *Exporter) WriteType(typ *Globals.Type) {
	if typ == nil {
		panic "typ == nil";
	}

	if typ.ref >= 0 {
		E.WriteTypeTag(-typ.ref);  // type already exported
		return;
	}

	if typ.form <= 0 {
		panic "typ.form <= 0";
	}
	E.WriteTypeTag(typ.form);
	typ.ref = E.type_ref;
	E.type_ref++;

	if typ.obj != nil {
		if typ.obj.typ != typ {
			panic "typ.obj.type() != typ";  // primary type
		}
		E.WriteString(typ.obj.ident);
		E.WritePackage(E.comp.pkgs[typ.obj.pnolev]);
	} else {
		E.WriteString("");
	}

	switch typ.form {
	case Type.ARRAY:
		E.WriteInt(typ.len_);
		E.WriteType(typ.elt);

	case Type.MAP:
		E.WriteType(typ.key);
		E.WriteType(typ.elt);

	case Type.CHANNEL:
		E.WriteInt(typ.flags);
		E.WriteType(typ.elt);

	case Type.FUNCTION:
		E.WriteInt(typ.flags);
		E.WriteScope(typ.scope);
		
	case Type.STRUCT, Type.INTERFACE:
		E.WriteScope(typ.scope);

	case Type.POINTER, Type.REFERENCE:
		E.WriteType(typ.elt);

	default:
		print "typ.form = ", typ.form, "\n";
		panic "UNREACHABLE";
	}
}


func (E *Exporter) WritePackage(pkg *Globals.Package) {
	if pkg.ref >= 0 {
		E.WritePackageTag(-pkg.ref);  // package already exported
		return;
	}

	if Object.PACKAGE <= 0 {
		panic "Object.PACKAGE <= 0";
	}
	E.WritePackageTag(Object.PACKAGE);
	pkg.ref = E.pkg_ref;
	E.pkg_ref++;

	E.WriteString(pkg.obj.ident);
	E.WriteString(pkg.file_name);
	E.WriteString(pkg.key);
}


func (E *Exporter) Export(comp* Globals.Compilation, file_name string) {
	if E.debug {
		print "exporting to ", file_name;
	}

	E.comp = comp;
	E.debug = true;
	E.pos = 0;
	E.pkg_ref = 0;
	E.type_ref = 0;
	
	// Predeclared types are "pre-exported".
	// TODO run the loop below only in debug mode
	{	i := 0;
		for p := Universe.types.first; p != nil; p = p.next {
			if p.typ.ref != i {
				panic "incorrect ref for predeclared type";
			}
			i++;
		}
	}
	E.type_ref = Universe.types.len_;
	
	pkg := comp.pkgs[0];
	E.WritePackage(pkg);
	for p := pkg.scope.entries.first; p != nil; p = p.next {
		if p.obj.exported {
			E.WriteObject(p.obj);
		}
	}
	E.WriteObjTag(0);

	if E.debug {
		print "\n(", E.pos, " bytes)\n";
	}
	
	data := string(E.buf)[0 : E.pos];
	ok := sys.writefile(file_name, data);
	
	if !ok {
		panic "export failed";
	}
}
