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
	i := len(s);
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
	return s + ".7"
}


func Import(C *Globals.Compilation, pkg_name string) (pno int) {
	panic "UNIMPLEMENTED";
}


func Export(C *Globals.Compilation) {
	file_name := FixExt(BaseName(C.src_name));  // strip src dir
	Export.Export(file_name/*, C */);
}


export Compile
func Compile(src_name string, verbose int) {
	comp := new(Globals.Compilation);
	comp.src_name = src_name;
	comp.pkg = nil;
	comp.nimports = 0;
	
	src, ok := sys.readfile(src_name);
	if !ok {
		print "cannot open ", src_name, "\n"
		return;
	}
	
	Universe.Init();

	S := new(Scanner.Scanner);
	S.Open(src_name, src);

	P := new(Parser.Parser);
	P.Open(S, verbose);
	
	print "parsing ", src_name, "\n";
	P.ParseProgram();
	//comp.Export();
}
