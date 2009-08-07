// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag";
	"fmt";
	"go/ast";
	"go/parser";
	"go/printer";
	"go/scanner";
	"io";
	"os";
	pathutil "path";
	"sort";
	"strings";
)


const pkgDir = "src/pkg";  // relative to $GOROOT


var (
	goroot = flag.String("goroot", os.Getenv("GOROOT"), "Go root directory");

	// operation modes
	allgo = flag.Bool("a", false, "include all .go files for package");
	comments = flag.Bool("c", false, "omit comments");
	silent = flag.Bool("s", false, "silent mode: parsing only");
	verbose = flag.Bool("v", false, "verbose mode: trace parsing");
	exports = flag.Bool("x", false, "show exports only");

	// layout control
	tabwidth = flag.Int("tabwidth", 8, "tab width");
	rawformat = flag.Bool("rawformat", false, "do not use a tabwriter");
	usespaces = flag.Bool("spaces", false, "align with blanks instead of tabs");
	optcommas = flag.Bool("optcommas", false, "print optional commas");
	optsemis = flag.Bool("optsemis", false, "print optional semicolons");
	reverse = flag.Bool("reverse", false, "print top-level declarations in reverse order without forward-declarations");
)


func usage() {
	fmt.Fprintf(os.Stderr, "usage: gofmt [flags] [file.go | pkgpath]\n");
	flag.PrintDefaults();
	os.Exit(2);
}


func parserMode() uint {
	mode := uint(0);
	if !*comments {
		mode |= parser.ParseComments;
	}
	if *verbose {
		mode |= parser.Trace;
	}
	return mode;
}


func isPkgFile(d *os.Dir) bool {
	// ignore non-Go files
	if !d.IsRegular() || strings.HasPrefix(d.Name, ".") || !strings.HasSuffix(d.Name, ".go") {
		return false;
	}

	// ignore test files unless explicitly included
	return *allgo || !strings.HasSuffix(d.Name, "_test.go");
}


func getPackage(path string) (*ast.Package, os.Error) {
	if len(path) == 0 {
		return nil, os.NewError("no path specified");
	}

	if strings.HasSuffix(path, ".go") || path == "/dev/stdin" {
		// single go file
		src, err := parser.ParseFile(path, nil, parserMode());
		if err != nil {
			return nil, err;
		}
		dirname, filename := pathutil.Split(path);
		return &ast.Package{src.Name.Value, dirname, map[string]*ast.File{filename: src}}, nil;
	}

	// len(path) > 0
	switch ch := path[0]; {
	case ch == '.':
		// cwd-relative path
		if cwd, err := os.Getwd(); err == nil {
			path = pathutil.Join(cwd, path);
		}
	case ch != '/':
		// goroot/pkgDir-relative path
		path = pathutil.Join(pathutil.Join(*goroot, pkgDir), path);
	}

	return parser.ParsePackage(path, isPkgFile, parserMode());
}


func printerMode() uint {
	mode := uint(0);
	if *rawformat {
		mode |= printer.RawFormat;
	}
	if *usespaces {
		mode |= printer.UseSpaces;
	}
	if *optcommas {
		mode |= printer.OptCommas;
	}
	if *optsemis {
		mode |= printer.OptSemis;
	}
	if *reverse {
		mode |= printer.Reverse;
	}
	return mode;
}


func main() {
	flag.Usage = usage;
	flag.Parse();

	path := "";
	switch flag.NArg() {
	case 0:
		path = "/dev/stdin";
	case 1:
		path = flag.Arg(0);
	default:
		usage();
	}

	pkg, err := getPackage(path);
	if err != nil {
		scanner.PrintError(os.Stderr, err);
		os.Exit(1);
	}

	if !*silent {
		if *exports {
			ast.PackageExports(pkg);
			_, err := printer.Fprint(os.Stdout, ast.MergePackageFiles(pkg), printerMode(), *tabwidth);
			if err != nil {
				fmt.Fprint(os.Stderr, err);
				os.Exit(2);
			}
		} else {
			for _, src := range pkg.Files {
				_, err := printer.Fprint(os.Stdout, src, printerMode(), *tabwidth);
				if err != nil {
					fmt.Fprint(os.Stderr, err);
					os.Exit(2);
				}
			}
		}
	}
}
