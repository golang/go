// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Type

import Globals "globals"
import Object "object"


export const /* form */ (
	// internal types
	// We should never see one of these.
	UNDEF = iota;
	
	// VOID types are used when we don't have a type. Never exported.
	// (exported type forms must be > 0)
	VOID;
	
	// BAD types are compatible with any type and don't cause further errors.
	// They are introduced only as a result of an error in the source code. A
	// correct program cannot have BAD types.
	BAD;
	
	// FORWARD types are forward-declared (incomplete) types. They can only
	// be used as element types of pointer types and must be resolved before
	// their internals are accessible.
	FORWARD;

	// TUPLE types represent multi-valued result types of functions and
	// methods.
	TUPLE;
	
	// The type of nil.
	NIL;

	// basic types
	BOOL; UINT; INT; FLOAT; STRING; INTEGER;
	
	// 'any' type  // TODO this should go away eventually
	ANY;
	
	// composite types
	ALIAS; ARRAY; STRUCT; INTERFACE; MAP; CHANNEL; FUNCTION; METHOD; POINTER;
)


export const /* Type.aux */ (
	SEND = 1;  // chan>
	RECV = 2;  // chan<
)


// The 'Type' declaration should be here as well, but 6g cannot handle
// this due to cross-package circular references. For now it's all in
// globals.go.


export func FormStr(form int) string {
	switch form {
	case VOID: return "VOID";
	case BAD: return "BAD";
	case FORWARD: return "FORWARD";
	case TUPLE: return "TUPLE";
	case NIL: return "NIL";
	case BOOL: return "BOOL";
	case UINT: return "UINT";
	case INT: return "INT";
	case FLOAT: return "FLOAT";
	case STRING: return "STRING";
	case ANY: return "ANY";
	case ALIAS: return "ALIAS";
	case ARRAY: return "ARRAY";
	case STRUCT: return "STRUCT";
	case INTERFACE: return "INTERFACE";
	case MAP: return "MAP";
	case CHANNEL: return "CHANNEL";
	case FUNCTION: return "FUNCTION";
	case METHOD: return "METHOD";
	case POINTER: return "POINTER";
	}
	return "<unknown Type form>";
}


export func Equal(x, y *Globals.Type) bool;

func Equal0(x, y *Globals.Type) bool {
	if x == y {
		return true;  // identical types are equal
	}

	if x.form == BAD || y.form == BAD {
		return true;  // bad types are always equal (avoid excess error messages)
	}

	// TODO where to check for *T == nil ?  
	if x.form != y.form {
		return false;  // types of different forms are not equal
	}

	switch x.form {
	case FORWARD, BAD:
		break;

	case NIL, BOOL, STRING, ANY:
		return true;

	case UINT, INT, FLOAT:
		return x.size == y.size;

	case ARRAY:
		return
			x.len == y.len &&
			Equal(x.elt, y.elt);

	case MAP:
		return
			Equal(x.key, y.key) &&
			Equal(x.elt, y.elt);

	case CHANNEL:
		return
			x.aux == y.aux &&
			Equal(x.elt, y.elt);

	case FUNCTION, METHOD:
		{	xp := x.scope.entries;
			yp := x.scope.entries;
			if	x.len != y.len &&  // number of parameters
				xp.len != yp.len  // recv + parameters + results
			{
				return false;
			}
			for p, q := xp.first, yp.first; p != nil; p, q = p.next, q.next {
				xf := p.obj;
				yf := q.obj;
				if xf.kind != Object.VAR || yf.kind != Object.VAR {
					panic("parameters must be vars");
				}
				if !Equal(xf.typ, yf.typ) {
					return false;
				}
			}
		}
		return true;

	case STRUCT:
		/*
		{	ObjList* xl = &x.scope.list;
			ObjList* yl = &y.scope.list;
			if xl.len() != yl.len() {
				return false;  // scopes of different sizes are not equal
			}
			for int i = xl.len(); i-- > 0; {
				Object* xf = (*xl)[i];
				Object* yf = (*yl)[i];
				ASSERT(xf.kind == Object.VAR && yf.kind == Object.VAR);
				if xf.name != yf.name) || ! EqualTypes(xf.type(), yf.type() {
					return false;
				}
			}
		}
		return true;
		*/
		// Scopes must be identical for them to be equal.
		// If we reach here, they weren't.
		return false;

	case INTERFACE:
		panic("UNIMPLEMENTED");
		return false;

	case POINTER:
		return Equal(x.elt, y.elt);
		
	case TUPLE:
		panic("UNIMPLEMENTED");
		return false;
	}

	panic("UNREACHABLE");
	return false;
}


export func Equal(x, y *Globals.Type) bool {
	res := Equal0(x, y);
	// TODO should do the check below only in debug mode
	if Equal0(y, x) != res {
		panic("type equality must be symmetric");
	}
	return res;
}


export func Assigneable(from, to *Globals.Type) bool {
	if Equal(from, to) {
		return true;
	}
	
	panic("UNIMPLEMENTED");
	return false;
}
