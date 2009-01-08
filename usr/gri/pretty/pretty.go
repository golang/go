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
	silent = Flag.Bool("s", false, nil, "silent mode: no pretty print output");
	verbose = Flag.Bool("v", false, &flags.verbose, "verbose mode: trace parsing");
	sixg = Flag.Bool("6g", true, &flags.sixg, "6g compatibility mode");
	//TODO fix this code again
	//deps = Flag.Bool("d", false, &flags.deps, "print dependency information only");
	columns = Flag.Bool("columns", Platform.USER == "gri", &flags.columns, "print column info in error messages");
	testmode = Flag.Bool("t", false, &flags.testmode, "test mode: interprets /* ERROR */ and /* SYNC */ comments");
	tokenchan = Flag.Bool("token_chan", false, &flags.tokenchan, "use token channel for scanner-parser connection");
)


func Usage() {
	print("usage: pretty { flags } { files }\n");
	Flag.PrintDefaults();
	sys.exit(0);
}


func main() {
	Flag.Parse();
	
	if Flag.NFlag() == 0 && Flag.NArg() == 0 {
		Usage();
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
			if !silent.BVal() && !flags.testmode {
				Printer.Print(prog);
			}
		}
	}
}
