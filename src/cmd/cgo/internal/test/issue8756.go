// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#cgo !darwin LDFLAGS: -lm
#include <math.h>
*/
import "C"
import (
	"testing"

	"cmd/cgo/internal/test/issue8756"
)

func test8756(t *testing.T) {
	issue8756.Pow()
	C.pow(1, 2)
}
