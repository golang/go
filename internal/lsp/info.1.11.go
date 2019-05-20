// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.12

package lsp

import (
	"fmt"
	"io"
)

func printBuildInfo(w io.Writer, verbose bool) {
	fmt.Fprintf(w, "no module information, gopls not built with go 1.11 or earlier\n")
}
