// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Object

import Globals "globals"


export BAD, CONST, TYPE, VAR, FUNC, PACKAGE
const /* kind */ (
	BAD = iota;  // error handling
	CONST; TYPE; VAR; FUNC; PACKAGE;
	PTYPE;  // primary type (import/export only)
)


export Object
type Object Globals.Object


export NewObject
func NewObject(kind int, name string) *Object {
	obj := new(Object);
	obj.mark = false;
	obj.kind = kind;
	obj.name = name;
	obj.type_ = nil;  // Universe::undef_t;
	obj.pnolev = 0;
	return obj;
}
