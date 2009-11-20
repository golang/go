// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes";
	"flag";
	"fmt";
	"go/ast";
	"go/parser";
	"go/printer";
	"go/scanner";
	"io";
	"os";
	pathutil "path";
	"strings";
)


var (
	// main operation modes
	list		= flag.Bool("l", false, "list files whose formatting differs from gofmt's");
	write		= flag.Bool("w", false, "write result to (source) file instead of stdout");
	rewriteRule	= flag.String("r", "", "rewrite rule (e.g., 'α[β:len(α)] -> α[β:]')");

	// debugging support
	comments	= flag.Bool("comments", true, "print comments");
	trace		= flag.Bool("trace", false, "print parse trace");

	// layout control
	tabwidth	= flag.Int("tabwidth", 8, "tab width");
	usespaces	= flag.Bool("spaces", false, "align with spaces instead of tabs");
)


var exitCode = 0
var rewrite func(*ast.File) *ast.File


func report(err os.Error) {
	scanner.PrintError(os.Stderr, err);
	exitCode = 2;
}


func usage() {
	fmt.Fprintf(os.Stderr, "usage: gofmt [flags] [path ...]\n");
	flag.PrintDefaults();
	os.Exit(2);
}


func parserMode() uint {
	mode := uint(0);
	if *comments {
		mode |= parser.ParseComments
	}
	if *trace {
		mode |= parser.Trace
	}
	return mode;
}


func printerMode() uint {
	mode := uint(0);
	if *usespaces {
		mode |= printer.UseSpaces
	}
	return mode;
}


func isGoFile(d *os.Dir) bool {
	// ignore non-Go files
	return d.IsRegular() && !strings.HasPrefix(d.Name, ".") && strings.HasSuffix(d.Name, ".go")
}


func processFile(filename string) os.Error {
	src, err := io.ReadFile(filename);
	if err != nil {
		return err
	}

	file, err := parser.ParseFile(filename, src, parserMode());
	if err != nil {
		return err
	}

	if rewrite != nil {
		file = rewrite(file)
	}

	var res bytes.Buffer;
	_, err = (&printer.Config{printerMode(), *tabwidth, nil}).Fprint(&res, file);
	if err != nil {
		return err
	}

	if bytes.Compare(src, res.Bytes()) != 0 {
		// formatting has changed
		if *list {
			fmt.Fprintln(os.Stdout, filename)
		}
		if *write {
			err = io.WriteFile(filename, res.Bytes(), 0);
			if err != nil {
				return err
			}
		}
	}

	if !*list && !*write {
		_, err = os.Stdout.Write(res.Bytes())
	}

	return err;
}


type fileVisitor chan os.Error

func (v fileVisitor) VisitDir(path string, d *os.Dir) bool {
	return true
}


func (v fileVisitor) VisitFile(path string, d *os.Dir) {
	if isGoFile(d) {
		v <- nil;	// synchronize error handler
		if err := processFile(path); err != nil {
			v <- err
		}
	}
}


func walkDir(path string) {
	// start an error handler
	v := make(fileVisitor);
	go func() {
		for err := range v {
			if err != nil {
				report(err)
			}
		}
	}();
	// walk the tree
	pathutil.Walk(path, v, v);
	close(v);
}


func main() {
	flag.Usage = usage;
	flag.Parse();
	if *tabwidth < 0 {
		fmt.Fprintf(os.Stderr, "negative tabwidth %d\n", *tabwidth);
		os.Exit(2);
	}

	initRewrite();

	if flag.NArg() == 0 {
		if err := processFile("/dev/stdin"); err != nil {
			report(err)
		}
	}

	for i := 0; i < flag.NArg(); i++ {
		path := flag.Arg(i);
		switch dir, err := os.Stat(path); {
		case err != nil:
			report(err)
		case dir.IsRegular():
			if err := processFile(path); err != nil {
				report(err)
			}
		case dir.IsDirectory():
			walkDir(path)
		}
	}

	os.Exit(exitCode);
}
