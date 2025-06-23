// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
   typedef struct S41761 S41761;
*/
import "C"

import (
	"cmd/cgo/internal/test/issue41761a"
	"testing"
)

func test41761(t *testing.T) {
	var x issue41761a.T
	_ = (*C.struct_S41761)(x.X)
}
