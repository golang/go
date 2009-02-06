// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os";
	"flag";
	Platform "platform";
	Printer "printer";
	Compilation "compilation";
)


var (
	flags Compilation.Flags;
	silent = flag.Bool("s", false, "silent mode: no pretty print output");
	html = flag.Bool("html", false, "generate html");
)

func init() {
	flag.BoolVar(&flags.Verbose, "v", false, "verbose mode: trace parsing");
	flag.BoolVar(&flags.Sixg, "6g", true, "6g compatibility mode");
	flag.BoolVar(&flags.Deps, "d", false, "print dependency information only");
	flag.BoolVar(&flags.Columns, "columns", Platform.USER == "gri", "print column info in error messages");
	flag.BoolVar(&flags.Testmode, "t", false, "test mode: interprets /* ERROR */ and /* SYNC */ comments");
}


func usage() {
	print("usage: pretty { flags } { files }\n");
	flag.PrintDefaults();
	sys.Exit(0);
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
			prog, nerrors := Compilation.Compile(src_file, &flags);
			if nerrors > 0 {
				if flags.Testmode {
					return;  // TODO we shouldn't need this
				}
				sys.Exit(1);
			}
			if !*silent && !flags.Testmode {
				Printer.Print(os.Stdout, *html, prog);
			}
		}
	}
}
