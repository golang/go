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


export Compile
func Compile(file_name string, verbose int) {
	src, ok := sys.readfile(file_name);
	if !ok {
		print "cannot open ", file_name, "\n"
		return;
	}
	
	Universe.Init();  // TODO eventually this should be only needed once
	
	comp := Globals.NewCompilation();

	scanner := new(Scanner.Scanner);
	scanner.Open(file_name, src);

	parser := new(Parser.Parser);
	parser.Open(comp, scanner, verbose);

	print "parsing ", file_name, "\n";
	parser.ParseProgram();
	if parser.S.nerrors > 0 {
		return;
	}
	
	// export
	exp := new(Export.Exporter);
	exp.Export(comp, Utils.FixExt(Utils.BaseName(file_name)));
}
