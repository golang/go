// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import OS "os"
import Platform "platform"
import Scanner "scanner"
import Parser "parser"
import AST "ast"


func assert(b bool) {
	if !b {
		panic("assertion failed");
	}
}


export type Flags struct {
	verbose bool;
	sixg bool;
	deps bool;
	columns bool;
	testmode bool;
	tokenchan bool;
}


export func Compile(src_file string, flags *Flags) (*AST.Program, int) {
	src, ok := Platform.ReadSourceFile(src_file);
	if !ok {
		print("cannot open ", src_file, "\n");
		return nil, 1;
	}

	var scanner Scanner.Scanner;
	scanner.Open(src_file, src, flags.columns, flags.testmode);

	var tstream *<-chan *Scanner.Token;
	if flags.tokenchan {
		tstream = scanner.TokenStream();
	}

	var parser Parser.Parser;
	parser.Open(flags.verbose, flags.sixg, flags.deps, &scanner, tstream);

	prog := parser.ParseProgram();
	return prog, scanner.nerrors;
}


func FileExists(name string) bool {
	fd, err := OS.Open(name, OS.O_RDONLY, 0);
	if err == nil {
		fd.Close();
		return true;
	}
	return false;
}


func AddDeps(globalset *map [string] bool, wset *AST.List, src_file string, flags *Flags) {
	dummy, found := globalset[src_file];
	if !found {
		globalset[src_file] = true;
		
		prog, nerrors := Compile(src_file, flags);
		if nerrors > 0 {
			return;
		}
		
		nimports := prog.decls.len();
		if nimports > 0 {
			print(src_file, ".6:\t");
			
			localset := new(map [string] bool);
			for i := 0; i < nimports; i++ {
				decl := prog.decls.at(i).(*AST.Decl);
				assert(decl.tok == Scanner.IMPORT && decl.val.tok == Scanner.STRING);
				src := decl.val.s;
				src = src[1 : len(src) - 1];  // strip "'s
				
				// ignore files when they are seen a 2nd time
				dummy, found := localset[src];
				if !found {
					localset[src] = true;
					if FileExists(src + ".go") {
						wset.Add(src);
						print(" ", src, ".6");
					} else if
						FileExists(Platform.GOROOT + "/pkg/" + src + ".6") ||
						FileExists(Platform.GOROOT + "/pkg/" + src + ".a") {
						
					} else {
						// TODO should collect these and print later
						//print("missing file: ", src, "\n");
					}
				}
			}
			print("\n\n");
		}
	}
}


export func ComputeDeps(src_file string, flags *Flags) {
	globalset := new(map [string] bool);
	wset := AST.NewList();
	wset.Add(src_file);
	for wset.len() > 0 {
		AddDeps(globalset, wset, wset.Pop().(string), flags);
	}
}
