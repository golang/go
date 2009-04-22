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
	"log";
	"os";
	"tabwriter";

	"astprinter";
)


var (
	columnsDefault bool;

	// operation modes
	columns = flag.Bool("columns", columnsDefault, "report column no. in error messages");
	silent = flag.Bool("s", false, "silent mode: no pretty print output");
	verbose = flag.Bool("v", false, "verbose mode: trace parsing");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("usetabs", false, "align with tabs instead of blanks");
)


func init() {
	user, err := os.Getenv("USER");
	columnsDefault = user == "gri";
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
	columns bool;
}


func (h *ErrorHandler) Error(pos token.Position, msg string) {
	// only report errors that are on a new line
	// in the hope to avoid most follow-up errors
	if pos.Line == h.lastline {
		return;
	}

	// report error
	fmt.Printf("%s:%d:", h.filename, pos.Line);
	if h.columns {
		fmt.Printf("%d:", pos.Column);
	}
	fmt.Printf(" %s\n", msg);
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

	// process files
	for i := 0; i < flag.NArg(); i++ {
		filename := flag.Arg(i);

		src, err := readFile(filename);
		if err != nil {
			log.Stderrf("ReadFile %s: %v", filename, err);
			continue;
		}

		prog, ok := parser.Parse(src, &ErrorHandler{filename, 0, false}, mode);
		if !ok {
			log.Stderr("Parse %s: syntax errors", filename);
			continue;
		}

		if !*silent {
			var printer astPrinter.Printer;
			writer := makeTabwriter(os.Stdout);
			printer.Init(writer, nil, nil /*prog.Comments*/, false);
			printer.DoProgram(prog);
			writer.Flush();
		}
	}
}
