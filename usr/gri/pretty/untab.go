// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os";
	"io";
	"flag";
	"fmt";
	"tabwriter";
)


var (
	tabwidth = flag.Int("tabwidth", 4, nil, "tab width");
	usetabs = flag.Bool("usetabs", false, nil, "align with tabs instead of blanks");
)


func Error(format string, params ...) {
	fmt.printf(format, params);
	sys.exit(1);
}


func Untab(name string, src *os.FD, dst *tabwriter.Writer) {
	n, err := io.Copy(src, dst);
	if err != nil {
		Error("error while processing %s (%v)", name, err);
	}
	//dst.Flush();
}


func main() {
	flag.Parse();
	padchar := byte(' ');
	if usetabs.BVal() {
		padchar = '\t';
	}
	dst := tabwriter.New(os.Stdout, int(tabwidth.IVal()), 1, padchar, true, false);
	if flag.NArg() > 0 {
		for i := 0; i < flag.NArg(); i++ {
			name := flag.Arg(i);
			src, err := os.Open(name, os.O_RDONLY, 0);
			if err != nil {
				Error("could not open %s (%v)\n", name, err);
			}
			Untab(name, src, dst);
			src.Close();  // ignore errors
		}
	} else {
		// no files => use stdin
		Untab("/dev/stdin", os.Stdin, dst);
	}
}
