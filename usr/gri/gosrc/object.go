// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Object

import Globals "globals"


export const /* kind */ (
	BAD = iota;  // error handling
	CONST; TYPE; VAR; FUNC; PACKAGE; LABEL;
	END;  // end of scope (import/export only)
)


// The 'Object' declaration should be here as well, but 6g cannot handle
// this due to cross-package circular references. For now it's all in
// globals.go.


export KindStr
func KindStr(kind int) string {
	switch kind {
	case BAD: return "BAD";
	case CONST: return "CONST";
	case TYPE: return "TYPE";
	case VAR: return "VAR";
	case FUNC: return "FUNC";
	case PACKAGE: return "PACKAGE";
	case LABEL: return "LABEL";
	case END: return "END";
	}
	return "<unknown Object kind>";
}
