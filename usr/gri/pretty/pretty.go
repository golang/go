// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Flag "flag"
import Platform "platform"
import Scanner "scanner"
import AST "ast"  // should not be needed
import Parser "parser"
import Printer "printer"


var (
    silent = Flag.Bool("s", false, nil, "silent mode: no pretty print output");
    verbose = Flag.Bool("v", false, nil, "verbose mode: trace parsing");
    sixg = Flag.Bool("6g", false, nil, "6g compatibility mode");
    tokenchan = Flag.Bool("token_chan", false, nil, "use token channel for scanner-parser connection");
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

	    src, ok := Platform.ReadSourceFile(src_file);
	    if !ok {
                print("cannot open ", src_file, "\n");
                return;
            }

	    print("- ", src_file, "\n");
	    scanner := new(Scanner.Scanner);
            scanner.Open(src_file, src);

	    var tstream *<-chan *Scanner.Token;
            if tokenchan.BVal() {
                tstream = scanner.TokenStream();
	    }

	    parser := new(Parser.Parser);
	    parser.Open(silent.BVal(), verbose.BVal(), scanner, tstream);

	    parser.ParseProgram();
	}
}
