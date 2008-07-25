// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Object

import Globals "globals"


export BAD, CONST, TYPE, VAR, FUNC, PACKAGE, LABEL, PTYPE
const /* kind */ (
	BAD = iota;  // error handling
	CONST; TYPE; VAR; FUNC; PACKAGE; LABEL;
	PTYPE;  // primary type (import/export only)
)


// The 'Object' declaration should be here as well, but 6g cannot handle
// this due to cross-package circular references. For now it's all in
// globals.go.
