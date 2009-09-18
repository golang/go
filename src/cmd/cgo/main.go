// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag";
	"fmt";
	"os";
	"tabwriter";
)

// Cgo; see gmp.go for an overview.

// TODO(rsc):
//	Emit correct line number annotations.
//	Make 6g understand the annotations.

func usage() {
	fmt.Fprint(os.Stderr, "usage: cgo file.cgo\n");
	flag.PrintDefaults();
}

func main() {
	flag.Usage = usage;
	flag.Parse();

	args := flag.Args();
	if len(args) != 1 {
		usage();
		os.Exit(2);
	}
	p := openProg(args[0]);
	p.loadDebugInfo();

	tw := tabwriter.NewWriter(os.Stdout, 1, 1, ' ', 0);
	for _, cref := range p.Crefs {
		what := "value";
		if cref.TypeName {
			what = "type";
		}
		fmt.Fprintf(tw, "%s:\t%s %s\tC %s\t%s\n", (*cref.Expr).Pos(), cref.Context, cref.Name, what, cref.DebugType);
	}
	tw.Flush();
}
