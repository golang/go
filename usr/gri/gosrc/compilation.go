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


export Compile
func Compile(comp *Globals.Compilation, file_name string) {
	src, ok := sys.readfile(file_name);
	if !ok {
		print "cannot open ", file_name, "\n"
		return;
	}
	
	scanner := new(Scanner.Scanner);
	scanner.Open(file_name, src);

	parser := new(Parser.Parser);
	parser.Open(comp, scanner);

	parser.ParseProgram();
	if parser.S.nerrors > 0 {
		return;
	}
	
	if !comp.flags.semantic_checks {
		return;
	}
	
	Verifier.Verify(comp);
	
	if comp.flags.print_export {
		Printer.PrintObject(comp, comp.pkg_list[0].obj, false);
	}
	
	Export.Export(comp, file_name);
}
