// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"astprinter";
	"flag";
	"fmt";
	"format";
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
	fmt.Fprintf(os.Stderr, "usage: pretty { flags } { files }\n");
	flag.PrintDefaults();
	os.Exit(1);
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
func makeTabwriter(writer io.Writer) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, 0);
}


func isValidPos(w io.Writer, env, value interface{}, name string) bool {
	pos := value.(token.Position);
	return pos.IsValid();
}


func isSend(w io.Writer, env, value interface{}, name string) bool {
	return value.(ast.ChanDir) & ast.SEND != 0;
}


func isRecv(w io.Writer, env, value interface{}, name string) bool {
	return value.(ast.ChanDir) & ast.RECV != 0;
}

func isMultiLineComment(w io.Writer, env, value interface{}, name string) bool {
	return value.([]byte)[1] == '*'
}


var fmap = format.FormatterMap{
	"isValidPos": isValidPos,
	"isSend": isSend,
	"isRecv": isRecv,
	"isMultiLineComment": isMultiLineComment,
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
		os.Exit(1);
	}
	ast_format, err := format.Parse(src, fmap);
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: format errors:\n%s", ast_txt, err);
		os.Exit(1);
	}

	// process files
	exitcode := 0;
	for i := 0; i < flag.NArg(); i++ {
		filename := flag.Arg(i);

		src, err := readFile(filename);
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
			if *formatter {
				var optSemi bool;  // formatting environment
				_, err := ast_format.Fprint(tw, &optSemi, prog);
				if err != nil {
					fmt.Fprintf(os.Stderr, "format error$$: %s", err);
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
