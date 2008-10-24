// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import Scanner "scanner"
import Parser "parser"
import AST "ast"



export type Flags struct {
	verbose bool;
	sixg bool;
	deps bool;
	columns bool;
	testmode bool;
	tokenchan bool;
}


type Compilation struct {
	prog *AST.Program;
	nerrors int;
}


export func Compile(src_file, src string, flags *Flags) *Compilation {
	var scanner Scanner.Scanner;
	scanner.Open(src_file, src, flags.columns, flags.testmode);

	var tstream *<-chan *Scanner.Token;
	if flags.tokenchan {
		tstream = scanner.TokenStream();
	}

	var parser Parser.Parser;
	parser.Open(flags.verbose, flags.sixg, &scanner, tstream);

	C := new(Compilation);
	C.prog = parser.ParseProgram();
	C.nerrors = scanner.nerrors;
	
	return C;
}
