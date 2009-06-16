// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag";
	"fmt";
	"go/parser";
	"go/printer";
	"io";
	"os";
	"sort";
	"tabwriter";
)


var (
	// operation modes
	silent = flag.Bool("s", false, "silent mode: parsing only");
	verbose = flag.Bool("v", false, "verbose mode: trace parsing");
	exports = flag.Bool("x", false, "show exports only");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("tabs", false, "align with tabs instead of blanks");
	optcommas = flag.Bool("optcommas", false, "print optional commas");
	optsemis = flag.Bool("optsemis", false, "print optional semicolons");
)


func usage() {
	fmt.Fprintf(os.Stderr, "usage: gofmt [flags] [file.go]\n");
	flag.PrintDefaults();
	os.Exit(1);
}


func parserMode() uint {
	mode := parser.ParseComments;
	if *verbose {
		mode |= parser.Trace;
	}
	return mode;
}


func printerMode() uint {
	mode := uint(0);
	if *exports {
		mode |= printer.ExportsOnly;
	}
	if *optcommas {
		mode |= printer.OptCommas;
	}
	if *optsemis {
		mode |= printer.OptSemis;
	}
	return mode;
}


func makeTabwriter(writer io.Writer) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, 0);
}


func main() {
	flag.Parse();

	var filename string;
	switch flag.NArg() {
	case 0: filename = "/dev/stdin";
	case 1: filename = flag.Arg(0);
	default: usage();
	}

	src, err := io.ReadFile(filename);
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", filename, err);
		os.Exit(1);
	}

	prog, err := parser.Parse(src, parserMode());
	if err != nil {
		if errors, ok := err.(parser.ErrorList); ok {
			sort.Sort(errors);
			for _, e := range errors {
				fmt.Fprintf(os.Stderr, "%s:%v\n", filename, e);
			}
		} else {
			fmt.Fprintf(os.Stderr, "%s: %v\n", filename, err);
		}
		os.Exit(1);
	}

	if !*silent {
		w := makeTabwriter(os.Stdout);
		printer.Fprint(w, prog, printerMode());
		w.Flush();
	}
}
