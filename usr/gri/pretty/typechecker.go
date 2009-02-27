// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package TypeChecker

import (
	AST "ast";
	Scanner "scanner";
)


type state struct {
	// setup
	err Scanner.ErrorHandler;
}


func (s *state) Init(err Scanner.ErrorHandler) {
	s.err = err;
}


// ----------------------------------------------------------------------------
// Support

func unimplemented() {
	panic("unimplemented");
}


func unreachable() {
	panic("unreachable");
}


func assert(pred bool) {
	if !pred {
		panic("assertion failed");
	}
}


func (s *state) Error(pos int, msg string) {
	s.err.Error(pos, msg);
}


// ----------------------------------------------------------------------------

func (s *state) CheckType() {
}


/*
func (s *state) CheckDeclaration(d *AST.Decl) {
	if d.Tok != Scanner.FUNC && d.List != nil {
		// group of parenthesized declarations
		for i := 0; i < d.List.Len(); i++ {
			s.CheckDeclaration(d.List.At(i).(*AST.Decl))
		}

	} else {
		// single declaration
		switch d.Tok {
		case Scanner.IMPORT:
		case Scanner.CONST:
		case Scanner.VAR:
		case Scanner.TYPE:
		case Scanner.FUNC:
		default:
			unreachable();
		}
	}
}
*/


func (s *state) CheckProgram(p *AST.Program) {
	for i := 0; i < len(p.Decls); i++ {
		//s.CheckDeclaration(p.Decls[i].(*AST.Decl));
	}
}


// ----------------------------------------------------------------------------

func CheckProgram(err Scanner.ErrorHandler, p *AST.Program) {
	var s state;
	s.Init(err);
	s.CheckProgram(p);
}
