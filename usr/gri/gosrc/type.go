// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Type

import Globals "globals"

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


const /* flag */ (
	SEND = 1 << iota;  // chan>
	RECV;  // chan< or method
)


export Type
type Type Globals.Type


export NewType
func NewType(form int) *Type {
	panic "UNIMPLEMENTED";
	return nil;
}
