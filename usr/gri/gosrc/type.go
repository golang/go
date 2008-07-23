// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Type

export
	UNDEF, BAD, NIL,
	BOOL, UINT, INT, FLOAT, STRING,
	ANY,
	ARRAY, STRUCT, INTERFACE, MAP, CHANNEL, FUNCTION, POINTER, REFERENCE

const /* form */ (
	// internal types
	UNDEF = iota; BAD; NIL;
	// basic types
	BOOL; UINT; INT; FLOAT; STRING;
	// 'any' type
	ANY;
	// composite types
	ARRAY; STRUCT; INTERFACE; MAP; CHANNEL; FUNCTION; POINTER; REFERENCE;
)


export
	SEND, RECV
	
const /* flag */ (
	SEND = 1 << iota;  // chan>
	RECV;  // chan< or method
)


// The 'Type' declaration should be here as well, but 6g cannot handle
// this due to cross-package circular references. For now it's all in
// globals.go.
