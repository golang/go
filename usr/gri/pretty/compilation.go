// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import (
	"vector";
	"utf8";
	"fmt";
	"os";
	Utils "utils";
	Platform "platform";
	"scanner";
	Parser "parser";
	AST "ast";
	TypeChecker "typechecker";
)


func assert(b bool) {
	if !b {
		panic("assertion failed");
	}
}


type Flags struct {
	Verbose bool;
	Deps bool;
	Columns bool;
}


type errorHandler struct {
	filename string;
	src []byte;
	columns bool;
	errline int;
	nerrors int;
}


func (h *errorHandler) Init(filename string, src []byte, columns bool) {
	h.filename = filename;
	h.src = src;
	h.columns = columns;
}


/*
// Compute (line, column) information for a given source position.
func (h *errorHandler) LineCol(pos int) (line, col int) {
	line = 1;
	lpos := 0;

	src := h.src;
	if pos > len(src) {
		pos = len(src);
	}

	for i := 0; i < pos; i++ {
		if src[i] == '\n' {
			line++;
			lpos = i;
		}
	}

	return line, utf8.RuneCount(src[lpos : pos]);
}
*/


func (h *errorHandler) ErrorMsg(loc scanner.Location, msg string) {
	fmt.Printf("%s:%d:", h.filename, loc.Line);
	if h.columns {
		fmt.Printf("%d:", loc.Col);
	}
	fmt.Printf(" %s\n", msg);

	h.errline = loc.Line;

	h.nerrors++;
	if h.nerrors >= 10 {
		sys.Exit(1);
	}
}


func (h *errorHandler) Error(loc scanner.Location, msg string) {
	// only report errors that are on a new line 
	// in the hope to avoid most follow-up errors
	if loc.Line != h.errline {
		h.ErrorMsg(loc, msg);
	}
}


func Compile(src_file string, flags *Flags) (*AST.Program, int) {
	src, ok := Platform.ReadSourceFile(src_file);
	if !ok {
		print("cannot open ", src_file, "\n");
		return nil, 1;
	}

	var err errorHandler;
	err.Init(src_file, src, flags.Columns);

	var scanner scanner.Scanner;
	scanner.Init(src, &err, true);

	var parser Parser.Parser;
	parser.Init(&scanner, &err, flags.Verbose);

	prog := parser.ParseProgram();

	if err.nerrors == 0 {
		TypeChecker.CheckProgram(&err, prog);
	}

	return prog, err.nerrors;
}


func fileExists(name string) bool {
	dir, err := os.Stat(name);
	return err == nil;
}

/*
func printDep(localset map [string] bool, wset *vector.Vector, decl AST.Decl2) {
	src := decl.Val.(*AST.BasicLit).Val;
	src = src[1 : len(src) - 1];  // strip "'s

	// ignore files when they are seen a 2nd time
	dummy, found := localset[src];
	if !found {
		localset[src] = true;
		if fileExists(src + ".go") {
			wset.Push(src);
			fmt.Printf(" %s.6", src);
		} else if
			fileExists(Platform.GOROOT + "/pkg/" + src + ".6") ||
			fileExists(Platform.GOROOT + "/pkg/" + src + ".a") {

		} else {
			// TODO should collect these and print later
			//print("missing file: ", src, "\n");
		}
	}
}
*/


func addDeps(globalset map [string] bool, wset *vector.Vector, src_file string, flags *Flags) {
	dummy, found := globalset[src_file];
	if !found {
		globalset[src_file] = true;

		prog, nerrors := Compile(src_file, flags);
		if nerrors > 0 {
			return;
		}

		nimports := len(prog.Decls);
		if nimports > 0 {
			fmt.Printf("%s.6:\t", src_file);

			localset := make(map [string] bool);
			for i := 0; i < nimports; i++ {
				decl := prog.Decls[i];
				panic();
				/*
				assert(decl.Tok == scanner.IMPORT);
				if decl.List == nil {
					printDep(localset, wset, decl);
				} else {
					for j := 0; j < decl.List.Len(); j++ {
						printDep(localset, wset, decl.List.At(j).(*AST.Decl));
					}
				}
				*/
			}
			print("\n\n");
		}
	}
}


func ComputeDeps(src_file string, flags *Flags) {
	panic("dependency printing currently disabled");
	globalset := make(map [string] bool);
	wset := vector.New(0);
	wset.Push(Utils.TrimExt(src_file, ".go"));
	for wset.Len() > 0 {
		addDeps(globalset, wset, wset.Pop().(string), flags);
	}
}
