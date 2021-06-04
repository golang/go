// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/typecheck"
	"cmd/internal/bio"
)

func WriteExports(out *bio.Writer) {
	// The linker also looks for the $$ marker - use char after $$ to distinguish format.
	out.WriteString("\n$$B\n") // indicate binary export format
	off := out.Offset()
	typecheck.WriteExports(out, true)
	size := out.Offset() - off
	out.WriteString("\n$$\n")

	if base.Debug.Export != 0 {
		fmt.Printf("BenchmarkExportSize:%s 1 %d bytes\n", base.Ctxt.Pkgpath, size)
	}
}
