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
import Export "export"


export Compilation
type Compilation struct {
  src_name string;
  pkg *Globals.Object;
  imports [256] *Package.Package;  // TODO need open arrays
  nimports int;
}


func (C *Compilation) Lookup(file_name string) *Package.Package {
	for i := 0; i < C.nimports; i++ {
		pkg := C.imports[i];
		if pkg.file_name == file_name {
			return pkg;
		}
	}
	return nil;
}


func (C *Compilation) Insert(pkg *Package.Package) {
	if C.Lookup(pkg.file_name) != nil {
		panic "package already inserted";
	}
	pkg.pno = C.nimports;
	C.imports[C.nimports] = pkg;
	C.nimports++;
}


func (C *Compilation) InsertImport(pkg *Package.Package) *Package.Package {
	p := C.Lookup(pkg.file_name);
	if (p == nil) {
		// no primary package found
		C.Insert(pkg);
		p = pkg;
	}
	return p;
}


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


func (C *Compilation) Import(pkg_name string) (pno int) {
	panic "UNIMPLEMENTED";
}


func (C *Compilation) Export() {
	file_name := FixExt(BaseName(C.src_name));  // strip src dir
	Export.Export(file_name/*, C */);
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
