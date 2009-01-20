// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	Flag "flag";
	Platform "platform";
	Printer "printer";
	Compilation "compilation";
)


var (
	flags Compilation.Flags;
	silent = Flag.Bool("s", false, "silent mode: no pretty print output");
)

func init() {
	Flag.BoolVar(&flags.Verbose, "v", false, "verbose mode: trace parsing");
	Flag.BoolVar(&flags.Sixg, "6g", true, "6g compatibility mode");
	//TODO fix this code again
	//Flag.BoolVar(&flags.Deps, "d", false, "print dependency information only");
	Flag.BoolVar(&flags.Columns, "columns", Platform.USER == "gri", "print column info in error messages");
	Flag.BoolVar(&flags.Testmode, "t", false, "test mode: interprets /* ERROR */ and /* SYNC */ comments");
	Flag.BoolVar(&flags.Tokenchan, "token_chan", false, "use token channel for scanner-parser connection");
}


func usage() {
	print("usage: pretty { flags } { files }\n");
	Flag.PrintDefaults();
	sys.Exit(0);
}


func main() {
	Flag.Parse();

	if Flag.NFlag() == 0 && Flag.NArg() == 0 {
		usage();
	}

	// process files
	for i := 0; i < Flag.NArg(); i++ {
		src_file := Flag.Arg(i);

		if false /* DISABLED flags.deps */ {
			Compilation.ComputeDeps(src_file, &flags);

		} else {
			prog, nerrors := Compilation.Compile(src_file, &flags);
			if nerrors > 0 {
				return;
			}
			if !*silent && !flags.Testmode {
				Printer.Print(prog);
			}
		}
	}
}
