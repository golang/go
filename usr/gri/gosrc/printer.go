// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Printer

import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"


type Printer struct {
	comp *Globals.Compilation;
	print_all bool;
	level int;
};


func (P *Printer) PrintObjectStruct(obj *Globals.Object);
func (P *Printer) PrintObject(obj *Globals.Object);

func (P *Printer) PrintTypeStruct(typ *Globals.Type);
func (P *Printer) PrintType(typ *Globals.Type);



func (P *Printer) Init(comp *Globals.Compilation, print_all bool) {
	P.comp = comp;
	P.print_all = print_all;
	P.level = 0;
}


func IsAnonymous(name string) bool {
	return len(name) == 0 || name[0] == '.';
}


func (P *Printer) PrintSigRange(typ *Globals.Type, a, b int) {
	scope := typ.scope;
	if a + 1 == b && IsAnonymous(scope.entries.ObjAt(a).ident) {
		P.PrintType(scope.entries.ObjAt(a).typ);  // result type only
	} else {
		print("(");
		for i := a; i < b; i++ {
			par := scope.entries.ObjAt(i);
			if i > a {
				print(", ");
			}
			print(par.ident, " ");
			P.PrintType(par.typ);
		}
		print(")");
	}
}


func (P *Printer) PrintSignature(typ *Globals.Type, fun *Globals.Object) {
	p0 := 0;
	if typ.form == Type.METHOD {
		p0 = 1;
	} else {
		if typ.form != Type.FUNCTION {
			panic("not a function or method");
		}
	}
	r0 := p0 + typ.len;
	l0 := typ.scope.entries.len;

	if P.level == 0 {
		print("func ");

		if 0 < p0 {
			P.PrintSigRange(typ, 0, p0);
			print(" ");
		}
	}

	if fun != nil {
		P.PrintObject(fun);
		//print(" ");
	} else if p0 > 0 {
		print(". ");
	}

	P.PrintSigRange(typ, p0, r0);

	if r0 < l0 {
		print(" ");
		P.PrintSigRange(typ, r0, l0);
	}
}


func (P *Printer) PrintIndent() {
	print("\n");
	for i := P.level; i > 0; i-- {
		print("\t");
	}
}


func (P *Printer) PrintScope(scope *Globals.Scope, delta int) {
	// determine the number of scope entries to print
	var n int;
	if P.print_all {
		n = scope.entries.len;
	} else {
		n = 0;
		for p := scope.entries.first; p != nil; p = p.next {
			if p.obj.exported && !IsAnonymous(p.obj.ident) {
				n++;
			}
		}
	}

	// print the scope
	const scale = 2;
	if n > 0 {
		P.level += delta;
		for p := scope.entries.first; p != nil; p = p.next {
			if P.print_all || p.obj.exported && !IsAnonymous(p.obj.ident) {
				P.PrintIndent();
				P.PrintObjectStruct(p.obj);
			}
		}
		P.level -= delta;
		P.PrintIndent();
	}
}


func (P *Printer) PrintObjectStruct(obj *Globals.Object) {
	switch obj.kind {
	case Object.BAD:
		P.PrintObject(obj);
		print(" /* bad */");

	case Object.CONST:
		print("const ");
		P.PrintObject(obj);
		print(" ");
		P.PrintType(obj.typ);

	case Object.TYPE:
		print("type ");
		P.PrintObject(obj);
		print(" ");
		P.PrintTypeStruct(obj.typ);

	case Object.VAR:
		print("var ");
		fallthrough;

	case Object.FIELD:
		P.PrintObject(obj);
		print(" ");
		P.PrintType(obj.typ);

	case Object.FUNC:
		P.PrintSignature(obj.typ, obj);

	case Object.BUILTIN:
		P.PrintObject(obj);
		print(" /* builtin */");

	case Object.PACKAGE:
		print("package ");
		P.PrintObject(obj);
		print(" ");
		P.PrintScope(P.comp.pkg_list[obj.pnolev].scope, 0);

	default:
		panic("UNREACHABLE");
	}

	if P.level > 0 {
		print(";");
	}
}


func (P *Printer) PrintObject(obj *Globals.Object) {
	if obj.pnolev > 0 {
		pkg := P.comp.pkg_list[obj.pnolev];
		if pkg.key == "" {
			// forward-declared package
			print(`"`, pkg.file_name, `"`);
		} else {
			// imported package
			print(pkg.obj.ident);
		}
		print(".");
	}
	print(obj.ident);
}


func (P *Printer) PrintTypeStruct(typ *Globals.Type) {
	switch typ.form {
	case Type.VOID:
		print("void");

	case Type.BAD:
		print("<bad type>");

	case Type.FORWARD:
		print("<forward type>");

	case Type.TUPLE:
		print("<tuple type>");

	case Type.NIL, Type.BOOL, Type.UINT, Type.INT, Type.FLOAT, Type.STRING, Type.ANY:
		if typ.obj == nil {
			panic("typ.obj == nil");
		}
		P.PrintType(typ);

	case Type.ALIAS:
		P.PrintType(typ.elt);
		if typ.key != typ.elt {
			print(" /* ");
			P.PrintType(typ.key);
			print(" */");
		}

	case Type.ARRAY:
		print("[]");
		P.PrintType(typ.elt);

	case Type.STRUCT:
		print("struct {");
		P.PrintScope(typ.scope, 1);
		print("}");

	case Type.INTERFACE:
		print("interface {");
		P.PrintScope(typ.scope, 1);
		print("}");

	case Type.MAP:
		print("map [");
		P.PrintType(typ.key);
		print("] ");
		P.PrintType(typ.elt);

	case Type.CHANNEL:
		switch typ.aux {
		case Type.SEND: print("chan <- ");
		case Type.RECV: print("<- chan ");
		case Type.SEND + Type.RECV: print("chan ");
		default: panic("UNREACHABLE");
		}
		P.PrintType(typ.elt);

	case Type.FUNCTION:
		P.PrintSignature(typ, nil);

	case Type.POINTER:
		print("*");
		P.PrintType(typ.elt);

	default:
		panic("UNREACHABLE");

	}
}


func (P *Printer) PrintType(typ *Globals.Type) {
	if typ.obj != nil {
		P.PrintObject(typ.obj);
	} else {
		P.PrintTypeStruct(typ);
	}
}


func PrintObject(comp *Globals.Compilation, obj *Globals.Object, print_all bool) {
	var P Printer;
	(&P).Init(comp, print_all);
	(&P).PrintObjectStruct(obj);
	print("\n");
}
