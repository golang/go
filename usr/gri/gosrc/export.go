// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Exporter

import Utils "utils"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"


type Exporter struct {
	comp *Globals.Compilation;
	debug bool;
	buf [4*1024] byte;
	buf_pos int;
	pkg_ref int;
	type_ref int;
};


func (E *Exporter) WriteObject(obj *Globals.Object);


func (E *Exporter) WriteByte(x byte) {
	E.buf[E.buf_pos] = x;
	E.buf_pos++;
	/*
	if E.debug {
		print " ", x;
	}
	*/
}


func (E *Exporter) WriteInt(x int) {
	x0 := x;
	for x < -64 || x >= 64 {
		E.WriteByte(byte(x & 127));
		x = int(uint(x >> 7));  // arithmetic shift
	}
	// -64 <= x && x < 64
	E.WriteByte(byte(x + 192));
	/*
	if E.debug {
		print " #", x0;
	}
	*/
}


func (E *Exporter) WriteString(s string) {
	n := len(s);
	E.WriteInt(n);
	for i := 0; i < n; i++ {
		E.WriteByte(s[i]);
	}
	if E.debug {
		print ` "`, s, `"`;
	}
}


func (E *Exporter) WritePackageTag(tag int) {
	E.WriteInt(tag);
	if E.debug {
		if tag >= 0 {
			print " [P", tag, "]";  // package ref
		} else {
			print "\nP", E.pkg_ref, ":";
		}
	}
}


func (E *Exporter) WriteTypeTag(tag int) {
	E.WriteInt(tag);
	if E.debug {
		if tag >= 0 {
			print " [T", tag, "]";  // type ref
		} else {
			print "\nT", E.type_ref, ": ", Type.FormStr(-tag);
		}
	}
}


func (E *Exporter) WriteObjectTag(tag int) {
	if tag < 0 {
		panic "tag < 0";
	}
	E.WriteInt(tag);
	if E.debug {
		print "\n", Object.KindStr(tag);
	}
}


func (E *Exporter) WritePackage(pkg *Globals.Package) {
	if E.comp.pkg_list[pkg.obj.pnolev] != pkg {
		panic "inconsistent package object"
	}

	if pkg.ref >= 0 {
		E.WritePackageTag(pkg.ref);  // package already exported
		return;
	}

	E.WritePackageTag(-1);
	pkg.ref = E.pkg_ref;
	E.pkg_ref++;

	E.WriteString(pkg.obj.ident);
	E.WriteString(pkg.file_name);
	E.WriteString(pkg.key);
}


func (E *Exporter) WriteScope(scope *Globals.Scope) {
	if E.debug {
		print " {";
	}

	for p := scope.entries.first; p != nil; p = p.next {
		E.WriteObject(p.obj);
	}
	E.WriteObject(nil);
	
	if E.debug {
		print " }";
	}
}


func (E *Exporter) WriteType(typ *Globals.Type) {
	if typ.ref >= 0 {
		E.WriteTypeTag(typ.ref);  // type already exported
		return;
	}

	if -typ.form >= 0 {
		panic "conflict with ref numbers";
	}
	E.WriteTypeTag(-typ.form);
	typ.ref = E.type_ref;
	E.type_ref++;

	// if we have a named type, export the type identifier and package
	ident := "";
	if typ.obj != nil {
		// named type
		if typ.obj.typ != typ {
			panic "inconsistent named type";
		}
		ident = typ.obj.ident;
		if !typ.obj.exported {
			// the type is invisible (it's identifier is not exported)
			// prepend "." to the identifier to make it an illegal
			// identifier and thus invisible in Go source code
			ident = "." + ident;
		}
	}
	
	E.WriteString(ident);
	if len(ident) > 0 {
		// named type
		E.WritePackage(E.comp.pkg_list[typ.obj.pnolev]);
	}
	
	switch typ.form {
	case Type.ALIAS:
		E.WriteType(typ.elt);

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
		panic "UNREACHABLE";
	}
}


func (E *Exporter) WriteObject(obj *Globals.Object) {
	if obj == nil {
		E.WriteObjectTag(Object.END);
		return;
	}
	E.WriteObjectTag(obj.kind);

	if obj.kind == Object.TYPE {
		// named types are handled entirely by WriteType()
		if obj.typ.obj != obj {
			panic "inconsistent named type"
		}
		E.WriteType(obj.typ);
		return;
	}

	E.WriteString(obj.ident);
	E.WriteType(obj.typ);

	switch obj.kind {
	case Object.CONST:
		E.WriteInt(0);  // should be the correct value

	case Object.VAR:
		E.WriteInt(0);  // should be the correct address/offset
		
	case Object.FUNC:
		E.WriteInt(0);  // should be the correct address/offset
		
	default:
		panic "UNREACHABLE";
	}
}


func (E *Exporter) Export(comp* Globals.Compilation, file_name string) {
	E.comp = comp;
	E.debug = comp.flags.debug;
	E.buf_pos = 0;
	E.pkg_ref = 0;
	E.type_ref = 0;
	
	if E.debug {
		print "exporting to ", file_name, "\n";
	}

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
	
	// export package 0
	pkg := comp.pkg_list[0];
	E.WritePackage(pkg);
	for p := pkg.scope.entries.first; p != nil; p = p.next {
		if p.obj.exported {
			E.WriteObject(p.obj);
		}
	}
	E.WriteObject(nil);
	
	if E.debug {
		print "\n(", E.buf_pos, " bytes)\n";
	}
	
	data := string(E.buf)[0 : E.buf_pos];
	ok := sys.writefile(file_name, data);
	
	if !ok {
		panic "export failed";
	}
}


export func Export(comp* Globals.Compilation, pkg_name string) {
	var E Exporter;
	(&E).Export(comp, Utils.FixExt(Utils.BaseName(pkg_name)));
}
