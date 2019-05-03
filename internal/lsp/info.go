// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.12

package lsp

import (
	"fmt"
	"io"
	"runtime/debug"
)

func printBuildInfo(w io.Writer, verbose bool) {
	if info, ok := debug.ReadBuildInfo(); ok {
		fmt.Fprintf(w, "%v\n", info.Path)
		printModuleInfo(w, &info.Main)
		if verbose {
			for _, dep := range info.Deps {
				printModuleInfo(w, dep)
			}
		}
	} else {
		fmt.Fprintf(w, "no module information, gopls not built in module mode\n")
	}
}

func printModuleInfo(w io.Writer, m *debug.Module) {
	fmt.Fprintf(w, "    %s@%s", m.Path, m.Version)
	if m.Sum != "" {
		fmt.Fprintf(w, " %s", m.Sum)
	}
	if m.Replace != nil {
		fmt.Fprintf(w, " => %v", m.Replace.Path)
	}
	fmt.Fprintf(w, "\n")
}
