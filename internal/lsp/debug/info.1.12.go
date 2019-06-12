// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.12

package debug

import (
	"fmt"
	"io"
	"runtime/debug"
)

func printBuildInfo(w io.Writer, verbose bool, mode PrintMode) {
	if info, ok := debug.ReadBuildInfo(); ok {
		fmt.Fprintf(w, "%v %v\n", info.Path, Version)
		printModuleInfo(w, &info.Main, mode)
		if verbose {
			for _, dep := range info.Deps {
				printModuleInfo(w, dep, mode)
			}
		}
	} else {
		fmt.Fprintf(w, "version %s, built in $GOPATH mode\n", Version)
	}
}

func printModuleInfo(w io.Writer, m *debug.Module, mode PrintMode) {
	fmt.Fprintf(w, "    %s@%s", m.Path, m.Version)
	if m.Sum != "" {
		fmt.Fprintf(w, " %s", m.Sum)
	}
	if m.Replace != nil {
		fmt.Fprintf(w, " => %v", m.Replace.Path)
	}
	fmt.Fprintf(w, "\n")
}
