// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag";
	"fmt";
	"go/ast";
	"go/parser";
	"go/token";
	"io";
	"os";
	"tabwriter";

	"astprinter";
	"format";
)


var (
	// operation modes
	columns bool;
	silent = flag.Bool("s", false, "silent mode: no pretty print output");
	verbose = flag.Bool("v", false, "verbose mode: trace parsing");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("tabs", false, "align with tabs instead of blanks");
	formatter = flag.Bool("formatter", false, "use formatter");  // TODO remove eventually
)


func init() {
	user, err := os.Getenv("USER");
	flag.BoolVar(&columns, "columns", user == "gri", "print column no. in error messages");
}


func usage() {
	print("usage: pretty { flags } { files }\n");
	flag.PrintDefaults();
	sys.Exit(0);
}


// TODO(gri) use library function for this once it exists
func readFile(filename string) ([]byte, os.Error) {
	f, err := os.Open(filename, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	defer f.Close();
	var b io.ByteBuffer;
	if n, err := io.Copy(f, &b); err != nil {
		return nil, err;
	}
	return b.Data(), nil;
}


// TODO(gri) move this function into tabwriter.go? (also used in godoc)
func makeTabwriter(writer io.Write) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, tabwriter.FilterHTML);
}


// TODO(gri) move this into parser as default handler
type ErrorHandler struct {
	filename string;
	lastline int;
}


func (h *ErrorHandler) Error(pos token.Position, msg string) {
	// only report errors that are on a new line
	// in the hope to avoid most follow-up errors
	if pos.Line == h.lastline {
		return;
	}
	h.lastline = pos.Line;

	// report error
	fmt.Fprintf(os.Stderr, "%s:%d:", h.filename, pos.Line);
	if columns {
		fmt.Fprintf(os.Stderr, "%d:", pos.Column);
	}
	fmt.Fprintf(os.Stderr, " %s\n", msg);
}


func main() {
	// handle flags
	flag.Parse();
	if flag.NFlag() == 0 && flag.NArg() == 0 {
		usage();
	}

	// determine parsing mode
	mode := parser.ParseComments;
	if *verbose {
		mode |= parser.Trace;
	}

	// get ast format
	const ast_txt = "ast.txt";
	src, err := readFile(ast_txt);
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", ast_txt, err);
		sys.Exit(1);
	}
	ast_format := format.Parse(src);
	if ast_format == nil {
		fmt.Fprintf(os.Stderr, "%s: format errors\n", ast_txt);
		sys.Exit(1);
	}

	// process files
	for i := 0; i < flag.NArg(); i++ {
		filename := flag.Arg(i);

		src, err := readFile(filename);
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", filename, err);
			continue;
		}

		prog, ok := parser.Parse(src, &ErrorHandler{filename, 0}, mode);

		if ok && !*silent {
			tw := makeTabwriter(os.Stdout);
			if *formatter {
				ast_format.Fprint(tw, prog);
			} else {
				var p astPrinter.Printer;
				p.Init(tw, nil, nil /*prog.Comments*/, false);
				p.DoProgram(prog);
			}
			tw.Flush();
		}
	}
}
