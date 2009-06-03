// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"astprinter";  // TODO remove once go/printer is fully functional
	"flag";
	"fmt";
	"go/ast";
	"go/parser";
	"go/token";
	"io";
	"os";
	"sort";
	"tabwriter";
)


var (
	// operation modes
	columns bool;
	// TODO remove silent flag eventually, can achieve same by proving no format file
	silent = flag.Bool("s", false, "silent mode: no pretty print output");
	verbose = flag.Bool("v", false, "verbose mode: trace parsing");

	// layout control
	format = flag.String("format", "", "format file");
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("tabs", false, "align with tabs instead of blanks");
)


func init() {
	user, err := os.Getenv("USER");
	flag.BoolVar(&columns, "columns", user == "gri", "print column no. in error messages");
}


func usage() {
	fmt.Fprintf(os.Stderr, "usage: pretty { flags } { files }\n");
	flag.PrintDefaults();
	os.Exit(1);
}


// TODO(gri) move this function into tabwriter.go? (also used in godoc)
func makeTabwriter(writer io.Writer) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, 0);
}


func main() {
	// handle flags
	flag.Parse();
	if flag.NFlag() == 0 && flag.NArg() == 0 {
		usage();
	}

	// initialize astFormat
	astFormat, err := ast.NewFormat(*format);
	if *format != "" && err != nil {  // ignore error if no format file given
		fmt.Fprintf(os.Stderr, "ast.NewFormat(%s): %v\n", *format, err);
		os.Exit(1);
	}

	// determine parsing mode
	mode := parser.ParseComments;
	if *verbose {
		mode |= parser.Trace;
	}

	// process files
	exitcode := 0;
	for i := 0; i < flag.NArg(); i++ {
		filename := flag.Arg(i);

		src, err := io.ReadFile(filename);
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", filename, err);
			exitcode = 1;
			continue;  // proceed with next file
		}

		prog, err := parser.Parse(src, mode);
		if err != nil {
			if errors, ok := err.(parser.ErrorList); ok {
				sort.Sort(errors);
				for _, e := range errors {
					fmt.Fprintf(os.Stderr, "%s:%v\n", filename, e);
				}
			} else {
				fmt.Fprintf(os.Stderr, "%s: %v\n", filename, err);
			}
			exitcode = 1;
			continue;  // proceed with next file
		}

		if !*silent {
			tw := makeTabwriter(os.Stdout);
			if *format != "" {
				_, err := astFormat.Fprint(tw, prog);
				if err != nil {
					fmt.Fprintf(os.Stderr, "format error: %v\n", err);
					exitcode = 1;
					continue;  // proceed with next file
				}
			} else {
				var p astPrinter.Printer;
				p.Init(tw, nil, nil /*prog.Comments*/, false);
				p.DoProgram(prog);
			}
			tw.Flush();
		}
	}

	os.Exit(exitcode);
}
