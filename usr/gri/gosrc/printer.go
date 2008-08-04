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
		P.PrintType(scope.entries.TypAt(a));  // result type only
	} else {
		print "(";
		for i := a; i < b; i++ {
			par := scope.entries.ObjAt(i);
			if i > a {
				print ", ";
			}
			print par.ident, " ";
			P.PrintType(par.typ);
		}
		print ")";
	}
}


func (P *Printer) PrintSignature(typ *Globals.Type, fun *Globals.Object) {
	if typ.form != Type.FUNCTION {
		panic "typ.form != Type.FUNCTION";
	}
	
	p0 := 0;
	if typ.flags & Type.RECV != 0 {
		p0 = 1;
	}
	r0 := p0 + typ.len_;
	l0 := typ.scope.entries.len_;
	
	if P.level == 0 {
		print "func ";

		if 0 < p0 {
			P.PrintSigRange(typ, 0, p0);
			print " ";
		}
	}
	
	if fun != nil {
		P.PrintObject(fun);
		//print " ";
	} else if p0 > 0 {
		print ". ";
	}
	
	P.PrintSigRange(typ, p0, r0);

	if r0 < l0 {
		print " ";
		P.PrintSigRange(typ, r0, l0);
	}
}


func (P *Printer) PrintIndent() {
	print "\n";
	for i := P.level; i > 0; i-- {
		print "\t";
	}
}


func (P *Printer) PrintScope(scope *Globals.Scope, delta int) {
	// determine the number of scope entries to print
	var n int;
	if P.print_all {
		n = scope.entries.len_;
	} else {
		n = 0;
		for p := scope.entries.first; p != nil; p = p.next {
			if p.obj.exported {
				n++;
			}
		}
	}
	
	// print the scope
	const scale = 2;
	if n > 0 {
		P.level += delta;
		for p := scope.entries.first; p != nil; p = p.next {
			if P.print_all || p.obj.exported {
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
		print "bad ";
		P.PrintObject(obj);

	case Object.CONST:
		print "const ";
		P.PrintObject(obj);
		print " ";
		P.PrintType(obj.typ);

	case Object.TYPE:
		print "type ";
		P.PrintObject(obj);
		print " ";
		P.PrintTypeStruct(obj.typ);

	case Object.VAR:
		if P.level == 0 {
			print "var ";
		}
		P.PrintObject(obj);
		print " ";
		P.PrintType(obj.typ);

	case Object.FUNC:
		P.PrintSignature(obj.typ, obj);

	case Object.PACKAGE:
		print "package ";
		P.PrintObject(obj);
		print " ";
		P.PrintScope(P.comp.pkg_list[obj.pnolev].scope, 0);

	default:
		panic "UNREACHABLE";
	}
	
	if P.level > 0 {
		print ";";
	}
}


func (P *Printer) PrintObject(obj *Globals.Object) {
	if obj.pnolev > 0 {
		print P.comp.pkg_list[obj.pnolev].obj.ident, ".";
	}
	print obj.ident;
}


func (P *Printer) PrintTypeStruct(typ *Globals.Type) {
	switch typ.form {
	case Type.UNDEF:
		print "<undef type>";

	case Type.BAD:
		print "<bad type>";

	case Type.NIL, Type.BOOL, Type.UINT, Type.INT, Type.FLOAT, Type.STRING, Type.ANY:
		if typ.obj == nil {
			panic "typ.obj == nil";
		}
		P.PrintType(typ);

	case Type.ALIAS:
		P.PrintType(typ.elt);

	case Type.ARRAY:
		print "[]";
		P.PrintType(typ.elt);

	case Type.STRUCT:
		print "struct {";
		P.PrintScope(typ.scope, 1);
		print "}";

	case Type.INTERFACE:
		print "interface {";
		P.PrintScope(typ.scope, 1);
		print "}";

	case Type.MAP:
		print "map [";
		P.PrintType(typ.key);
		print "] ";
		P.PrintType(typ.elt);

	case Type.CHANNEL:
		print "chan";
		switch typ.flags {
		case Type.SEND: print " -<";
		case Type.RECV: print " <-";
		case Type.SEND + Type.RECV:  // nothing to print
		default: panic "UNREACHABLE";
		}
		print " ";
		P.PrintType(typ.elt);

	case Type.FUNCTION:
		P.PrintSignature(typ, nil);

	case Type.POINTER:
		print "*";
		P.PrintType(typ.elt);

	case Type.REFERENCE:
		print "&";
		P.PrintType(typ.elt);

	default:
		panic "UNREACHABLE";
		
	}
}


func (P *Printer) PrintType(typ *Globals.Type) {
	if typ.obj != nil {
		P.PrintObject(typ.obj);
	} else {
		P.PrintTypeStruct(typ);
	}
}


export PrintObject
func PrintObject(comp *Globals.Compilation, obj *Globals.Object, print_all bool) {
	var P Printer;
	(&P).Init(comp, print_all);
	(&P).PrintObjectStruct(obj);
}
