// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Objwriter reads an object file description in an unspecified format
// and writes a Go object file. It is invoked by parts of the toolchain
// that have not yet been converted from C to Go and should not be
// used otherwise.
package main

import "cmd/internal/obj"
import (
	"cmd/internal/obj/x86"
)

// TODO(rsc): Implement.
// For now we just check that the objwriter binary is available to be run.

func main() {
	_ = obj.Exported
	_ = x86.Exported
}
