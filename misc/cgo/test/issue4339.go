// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include "issue4339.h"
*/
import "C"

import "testing"

func test4339(t *testing.T) {
	C.handle4339(&C.exported4339)
}
