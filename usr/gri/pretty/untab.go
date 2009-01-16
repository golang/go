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
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("usetabs", false, "align with tabs instead of blanks");
)


func error(format string, params ...) {
	fmt.Printf(format, params);
	sys.Exit(1);
}


func untab(name string, src *os.FD, dst *tabwriter.Writer) {
	n, err := io.Copy(src, dst);
	if err != nil {
		error("error while processing %s (%v)", name, err);
	}
	//dst.Flush();
}


func main() {
	flag.Parse();
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	dst := tabwriter.New(os.Stdout, *tabwidth, 1, padchar, true, false);
	if flag.NArg() > 0 {
		for i := 0; i < flag.NArg(); i++ {
			name := flag.Arg(i);
			src, err := os.Open(name, os.O_RDONLY, 0);
			if err != nil {
				error("could not open %s (%v)\n", name, err);
			}
			untab(name, src, dst);
			src.Close();  // ignore errors
		}
	} else {
		// no files => use stdin
		untab("/dev/stdin", os.Stdin, dst);
	}
}
