// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import Globals "globals"
import Object "object"
import Type "type"
import Package "package"
import Scanner "scanner"
import Parser "parser"


export Compilation
type Compilation struct {
  src_name string;
  pkg *Globals.Object;
  imports *Globals.List;  // a list of *Globals.Package
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
func Compile() {
}
