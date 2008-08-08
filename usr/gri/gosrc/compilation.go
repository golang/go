// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import Utils "utils"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"
import Scanner "scanner"
import AST "ast"
import Parser "parser"
import Export "export"
import Printer "printer"
import Verifier "verifier"


export func Compile(flags *Globals.Flags, filename string) {
	// setup compilation
	comp := new(Globals.Compilation);
	comp.flags = flags;
	comp.Compile = &Compile;
	
	src, ok := sys.readfile(filename);
	if !ok {
		print "cannot open ", filename, "\n"
		return;
	}
	
	print filename, "\n";
	
	scanner := new(Scanner.Scanner);
	scanner.Open(filename, src);
	
	var tstream *chan *Scanner.Token;
	if comp.flags.token_chan {
		tstream = new(chan *Scanner.Token, 100);
		go scanner.Server(tstream);
	}

	parser := new(Parser.Parser);
	parser.Open(comp, scanner, tstream);

	parser.ParseProgram();
	if parser.S.nerrors > 0 {
		return;
	}
	
	if !comp.flags.ast {
		return;
	}
	
	Verifier.Verify(comp);
	
	if comp.flags.print_interface {
		Printer.PrintObject(comp, comp.pkg_list[0].obj, false);
	}
	
	Export.Export(comp, filename);
}
