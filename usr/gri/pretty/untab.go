// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	OS "os";
	IO "io";
	Flag "flag";
	Fmt "fmt";
	TabWriter "tabwriter";
)


var (
	tabwidth = Flag.Int("tabwidth", 4, nil, "tab width");
)


func Error(fmt string, params ...) {
	Fmt.printf(fmt, params);
	sys.exit(1);
}


func Untab(name string, src *OS.FD, dst *TabWriter.TabWriter) {
	n, err := IO.Copyn(src, dst, 2e9 /* inf */);  // TODO use Copy
	if err != nil {
		Error("error while processing %s (%v)", name, err);
	}
	//dst.Flush();
}


func main() {
	Flag.Parse();
	dst := TabWriter.MakeTabWriter(OS.Stdout, int(tabwidth.IVal()));
	if Flag.NArg() > 0 {
		for i := 0; i < Flag.NArg(); i++ {
			name := Flag.Arg(i);
			src, err := OS.Open(name, OS.O_RDONLY, 0);
			if err != nil {
				Error("could not open %s (%v)\n", name, err);
			}
			Untab(name, src, dst);
			src.Close();  // ignore errors
		}
	} else {
		// no files => use stdin
		Untab("/dev/stdin", OS.Stdin, dst);
	}
}
