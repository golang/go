// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import (
	"vector";
	"utf8";
	"fmt";
	"os";
	"utils";
	"platform";
	"token";
	"scanner";
	"parser";
	"ast";
	"typechecker";
	"sort";
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


type Error struct {
	Pos token.Position;
	Msg string;
}


type ErrorList []Error

func (list ErrorList) Len() int { return len(list); }
func (list ErrorList) Less(i, j int) bool { return list[i].Pos.Offset < list[j].Pos.Offset; }
func (list ErrorList) Swap(i, j int) { list[i], list[j] = list[j], list[i]; }


type errorHandler struct {
	filename string;
	columns bool;
	errline int;
	errors vector.Vector;
}


func (h *errorHandler) Init(filename string, columns bool) {
	h.filename = filename;
	h.columns = columns;
	h.errors.Init(0);
}


func (h *errorHandler) Error(pos token.Position, msg string) {
	// only report errors that are on a new line 
	// in the hope to avoid most follow-up errors
	if pos.Line == h.errline {
		return;
	}

	// report error
	fmt.Printf("%s:%d:", h.filename, pos.Line);
	if h.columns {
		fmt.Printf("%d:", pos.Column);
	}
	fmt.Printf(" %s\n", msg);

	// collect the error
	h.errors.Push(Error{pos, msg});
	h.errline = pos.Line;
}


func Compile(filename string, flags *Flags) (*ast.Program, ErrorList) {
	src, os_err := os.Open(filename, os.O_RDONLY, 0);
	defer src.Close();
	if os_err != nil {
		fmt.Printf("cannot open %s (%s)\n", filename, os_err.String());
		return nil, nil;
	}

	var err errorHandler;
	err.Init(filename, flags.Columns);

	mode := parser.ParseComments;
	if flags.Verbose {
		mode |= parser.Trace;
	}
	prog, ok2 := parser.Parse(src, &err, mode);

	if ok2 {
		TypeChecker.CheckProgram(&err, prog);
	}
	
	// convert error list and sort it
	errors := make(ErrorList, err.errors.Len());
	for i := 0; i < err.errors.Len(); i++ {
		errors[i] = err.errors.At(i).(Error);
	}
	sort.Sort(errors);

	return prog, errors;
}


func fileExists(name string) bool {
	dir, err := os.Stat(name);
	return err == nil;
}

/*
func printDep(localset map [string] bool, wset *vector.Vector, decl ast.Decl2) {
	src := decl.Val.(*ast.BasicLit).Val;
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

		prog, errors := Compile(src_file, flags);
		if errors == nil || len(errors) > 0 {
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
						printDep(localset, wset, decl.List.At(j).(*ast.Decl));
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
