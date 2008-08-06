// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Type

import Globals "globals"
import Object "object"


export const /* form */ (
	// internal types
	UNDEF = iota; VOID; BAD; NIL;
	// basic types
	BOOL; UINT; INT; FLOAT; STRING; INTEGER;
	// 'any' type
	ANY;
	// composite types
	ALIAS; ARRAY; STRUCT; INTERFACE; MAP; CHANNEL; FUNCTION; POINTER; REFERENCE;
)


export const /* flag */ (
	SEND = 1 << iota;  // chan>
	RECV;  // chan< or method
)


// The 'Type' declaration should be here as well, but 6g cannot handle
// this due to cross-package circular references. For now it's all in
// globals.go.


export func FormStr(form int) string {
	switch form {
	case UNDEF: return "UNDEF";
	case VOID: return "VOID";
	case BAD: return "BAD";
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
	case POINTER: return "POINTER";
	case REFERENCE: return "REFERENCE";
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
	case UNDEF, BAD:
		break;

	case NIL, BOOL, STRING, ANY:
		return true;

	case UINT, INT, FLOAT:
		return x.size == y.size;

	case ARRAY:
		return
			x.len_ == y.len_ &&
			Equal(x.elt, y.elt);

	case MAP:
		return
			Equal(x.aux, y.aux) &&
			Equal(x.elt, y.elt);

	case CHANNEL:
		return
			x.flags == y.flags &&
			Equal(x.elt, y.elt);

	case FUNCTION:
		{	xp := x.scope.entries;
			yp := x.scope.entries;
			if	x.flags != y.flags &&  // function or method
				x.len_ != y.len_ &&  // number of parameters
				xp.len_ != yp.len_  // recv + parameters + results
			{
				return false;
			}
			for p, q := xp.first, yp.first; p != nil; p, q = p.next, q.next {
				xf := p.obj;
				yf := q.obj;
				if xf.kind != Object.VAR || yf.kind != Object.VAR {
					panic "parameters must be vars";
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
		panic "UNIMPLEMENTED";
		return false;

	case POINTER, REFERENCE:
		return Equal(x.elt, y.elt);
	}

	panic "UNREACHABLE";
	return false;
}


export func Equal(x, y *Globals.Type) bool {
	res := Equal0(x, y);
	// TODO should do the check below only in debug mode
	if Equal0(y, x) != res {
		panic "type equality must be symmetric";
	}
	return res;
}
