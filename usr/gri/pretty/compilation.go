// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import (
	"array";
	"utf8";
	OS "os";
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


export type Flags struct {
	verbose bool;
	sixg bool;
	deps bool;
	columns bool;
	testmode bool;
	tokenchan bool;
}


type ErrorHandler struct {
	filename string;
	src string;
	nerrors int;
	nwarnings int;
	errpos int;
	columns bool;
}


func (h *ErrorHandler) Init(filename, src string, columns bool) {
	h.filename = filename;
	h.src = src;
	h.nerrors = 0;
	h.nwarnings = 0;
	h.errpos = 0;
	h.columns = columns;
}


// Compute (line, column) information for a given source position.
func (h *ErrorHandler) LineCol(pos int) (line, col int) {
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

	return line, utf8.RuneCountInString(src, lpos, pos - lpos);
}


func (h *ErrorHandler) ErrorMsg(pos int, msg string) {
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
		sys.exit(1);
	}
}


func (h *ErrorHandler) Error(pos int, msg string) {
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


func (h *ErrorHandler) Warning(pos int, msg string) {
	panic("UNIMPLEMENTED");
}


export func Compile(src_file string, flags *Flags) (*AST.Program, int) {
	src, ok := Platform.ReadSourceFile(src_file);
	if !ok {
		print("cannot open ", src_file, "\n");
		return nil, 1;
	}

	var err ErrorHandler;
	err.Init(src_file, src, flags.columns);

	var scanner Scanner.Scanner;
	scanner.Init(&err, src, true, flags.testmode);

	var tstream <-chan *Scanner.Token;
	if flags.tokenchan {
		tstream = scanner.TokenStream();
	}

	var parser Parser.Parser;
	parser.Open(flags.verbose, flags.sixg, flags.deps, &scanner, tstream);

	prog := parser.ParseProgram();

	if err.nerrors == 0 {
		TypeChecker.CheckProgram(&err, prog);
	}

	return prog, err.nerrors;
}


func FileExists(name string) bool {
	fd, err := OS.Open(name, OS.O_RDONLY, 0);
	if err == nil {
		fd.Close();
		return true;
	}
	return false;
}


func AddDeps(globalset map [string] bool, wset *array.Array, src_file string, flags *Flags) {
	dummy, found := globalset[src_file];
	if !found {
		globalset[src_file] = true;

		prog, nerrors := Compile(src_file, flags);
		if nerrors > 0 {
			return;
		}

		nimports := prog.decls.Len();
		if nimports > 0 {
			print(src_file, ".6:\t");

			localset := make(map [string] bool);
			for i := 0; i < nimports; i++ {
				decl := prog.decls.At(i).(*AST.Decl);
				assert(decl.tok == Scanner.IMPORT && decl.val.tok == Scanner.STRING);
				src := decl.val.obj.ident;
				src = src[1 : len(src) - 1];  // strip "'s

				// ignore files when they are seen a 2nd time
				dummy, found := localset[src];
				if !found {
					localset[src] = true;
					if FileExists(src + ".go") {
						wset.Push(src);
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
	globalset := make(map [string] bool);
	wset := array.New(0);
	wset.Push(src_file);
	for wset.Len() > 0 {
		AddDeps(globalset, wset, wset.Pop().(string), flags);
	}
}
