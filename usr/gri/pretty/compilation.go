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
	Scanner "scanner";
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
	nerrors int;
	nwarnings int;
	errpos int;
	columns bool;
}


func (h *errorHandler) Init(filename string, src []byte, columns bool) {
	h.filename = filename;
	h.src = src;
	h.nerrors = 0;
	h.nwarnings = 0;
	h.errpos = 0;
	h.columns = columns;
}


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


func (h *errorHandler) ErrorMsg(pos int, msg string) {
	print(h.filename, ":");
	if pos >= 0 {
		// print position
		line, col := h.LineCol(pos);
		print(line, ":");
		if h.columns {
			print(col, ":");
		}
	}
	print(" ", msg, "\n");

	h.nerrors++;
	h.errpos = pos;

	if h.nerrors >= 10 {
		sys.Exit(1);
	}
}


func (h *errorHandler) Error(pos int, msg string) {
	// only report errors that are sufficiently far away from the previous error
	// in the hope to avoid most follow-up errors
	const errdist = 20;
	delta := pos - h.errpos;  // may be negative!
	if delta < 0 {
		delta = -delta;
	}

	if delta > errdist || h.nerrors == 0 /* always report first error */ {
		h.ErrorMsg(pos, msg);
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

	var scanner Scanner.Scanner;
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
	fd, err := os.Open(name, os.O_RDONLY, 0);
	defer fd.Close();
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
				assert(decl.Tok == Scanner.IMPORT);
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
