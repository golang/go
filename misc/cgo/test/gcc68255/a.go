// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that it's OK to have C code that does nothing other than
// initialize a global variable.  This used to fail with gccgo.

package gcc68255

/*
#include "c.h"
*/
import "C"

func F() bool {
	return C.v != nil
}
