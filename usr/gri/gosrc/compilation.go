// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"
import Scanner "scanner"
import Parser "parser"
import Export "export"


func BaseName(s string) string {
	// TODO this is not correct for non-ASCII strings!
	i := len(s) - 1;
	for i >= 0 && s[i] != '/' {
		if s[i] > 128 {
			panic "non-ASCII string"
		}
		i--;
	}
	return s[i + 1 : len(s)];
}


func FixExt(s string) string {
	i := len(s) - 3;  // 3 == len(".go");
	if s[i : len(s)] == ".go" {
		s = s[0 : i];
	}
	return s + ".7";
}


export Compile
func Compile(file_name string, verbose int) {
	src, ok := sys.readfile(file_name);
	if !ok {
		print "cannot open ", file_name, "\n"
		return;
	}
	
	Universe.Init();  // TODO eventually this should be only needed once
	
	comp := Globals.NewCompilation();
	pkg := Globals.NewPackage(file_name);
	comp.Insert(pkg);
	if comp.npkgs != 1 {
		panic "should have exactly one package now";
	}

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
	export_file_name := FixExt(BaseName(file_name));  // strip file dir
	Export.Export(comp, export_file_name);
}
