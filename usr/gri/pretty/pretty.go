// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os";
	"flag";
	"platform";
	"compilation";
	"tabwriter";
	"ast";
	"astprinter";
)


var (
	flags Compilation.Flags;
	silent = flag.Bool("s", false, "silent mode: no pretty print output");

	// layout control
	html = flag.Bool("html", false, "generate html");
	tabwidth = flag.Int("pretty_tabwidth", 4, "tab width");
	usetabs = flag.Bool("pretty_usetabs", false, "align with tabs instead of blanks");
)


func init() {
	flag.BoolVar(&flags.Verbose, "v", false, "verbose mode: trace parsing");
	flag.BoolVar(&flags.Deps, "d", false, "print dependency information only");
	flag.BoolVar(&flags.Columns, "columns", Platform.USER == "gri", "print column info in error messages");
}


func usage() {
	print("usage: pretty { flags } { files }\n");
	flag.PrintDefaults();
	sys.Exit(0);
}


func print(prog *ast.Program) {
	// initialize tabwriter for nicely aligned output
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	writer := tabwriter.NewWriter(os.Stdout, *tabwidth, 1, padchar, tabwriter.FilterHTML);

	// initialize printer
	var printer astPrinter.Printer;
	printer.Init(writer, prog.Comments, *html);

	printer.DoProgram(prog);

	// flush any pending output
	writer.Flush();
}


func main() {
	flag.Parse();

	if flag.NFlag() == 0 && flag.NArg() == 0 {
		usage();
	}

	// process files
	for i := 0; i < flag.NArg(); i++ {
		src_file := flag.Arg(i);

		if flags.Deps {
			Compilation.ComputeDeps(src_file, &flags);

		} else {
			prog, errors := Compilation.Compile(src_file, &flags);
			if errors == nil || len(errors) > 0 {
				sys.Exit(1);
			}
			if !*silent {
				print(prog);
			}
		}
	}
}
