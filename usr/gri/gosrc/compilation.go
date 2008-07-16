// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"
import Package "package"
import Scanner "scanner"
import Parser "parser"


export Compilation
type Compilation struct {
  src_name string;
  pkg *Globals.Object;
  imports [256] *Package.Package;  // TODO need open arrays
  nimports int;
}


func (C *Compilation) Lookup(pkg_name string) *Package.Package {
	panic "UNIMPLEMENTED";
	return nil;
}


func (C *Compilation) Insert(pkg *Package.Package) {
	panic "UNIMPLEMENTED";
}


func (C *Compilation) InsertImport(pkg *Package.Package) *Package.Package {
	panic "UNIMPLEMENTED";
	return nil;
}


func (C *Compilation) Import(pkg_name string) (pno int) {
	panic "UNIMPLEMENTED";
}


func (C *Compilation) Export() {
	panic "UNIMPLEMENTED";
}


export Compile
func Compile(src_name string, verbose int) {
	comp := new(Compilation);
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
}
