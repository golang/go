// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"fmt"
	"io"

	"cmd/compile/internal/base"
	"cmd/compile/internal/typecheck"
	"cmd/internal/bio"
)

// writeNewExportFunc is a hook that can be added to append extra
// export data after the normal export data section. It allows
// experimenting with new export data format designs without requiring
// immediate support in the go/internal or x/tools importers.
var writeNewExportFunc func(out io.Writer)

func WriteExports(out *bio.Writer) {
	// When unified IR exports are enable, we simply append it to the
	// end of the normal export data (with compiler extensions
	// disabled), and write an extra header giving its size.
	//
	// If the compiler sees this header, it knows to read the new data
	// instead; meanwhile the go/types importers will silently ignore it
	// and continue processing the old export instead.
	//
	// This allows us to experiment with changes to the new export data
	// format without needing to update the go/internal/gcimporter or
	// (worse) x/tools/go/gcexportdata.

	useNewExport := writeNewExportFunc != nil

	var old, new bytes.Buffer

	typecheck.WriteExports(&old, !useNewExport)

	if useNewExport {
		writeNewExportFunc(&new)
	}

	oldLen := old.Len()
	newLen := new.Len()

	if useNewExport {
		fmt.Fprintf(out, "\nnewexportsize %v\n", newLen)
	}

	// The linker also looks for the $$ marker - use char after $$ to distinguish format.
	out.WriteString("\n$$B\n") // indicate binary export format
	io.Copy(out, &old)
	out.WriteString("\n$$\n")
	io.Copy(out, &new)

	if base.Debug.Export != 0 {
		fmt.Printf("BenchmarkExportSize:%s 1 %d bytes\n", base.Ctxt.Pkgpath, oldLen)
		if useNewExport {
			fmt.Printf("BenchmarkNewExportSize:%s 1 %d bytes\n", base.Ctxt.Pkgpath, newLen)
		}
	}
}
