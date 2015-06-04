// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the canonical method checker.

// This file contains the code to check canonical methods.

package testdata

import (
	"fmt"
)

type MethodTest int

func (t *MethodTest) Scan(x fmt.ScanState, c byte) { // ERROR "should have signature Scan"
}

type MethodTestInterface interface {
	ReadByte() byte // ERROR "should have signature ReadByte"
}
