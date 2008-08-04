// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Type

export const /* form */ (
	// internal types
	UNDEF = iota; BAD; NIL;
	// basic types
	BOOL; UINT; INT; FLOAT; STRING;
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
