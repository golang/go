// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package TypeChecker

import (
	AST "ast";
	Scanner "scanner";
	Universe "universe";
)


type State struct {
	// setup
	err Scanner.ErrorHandler;
}


func (s *State) Init(err Scanner.ErrorHandler) {
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


func (s *State) Error(pos int, msg string) {
	s.err.Error(pos, msg);
}


// ----------------------------------------------------------------------------

func (s *State) CheckType() {
}


func (s *State) CheckDeclaration(d *AST.Decl) {
	if d.tok != Scanner.FUNC && d.list != nil {
		// group of parenthesized declarations
		for i := 0; i < d.list.Len(); i++ {
			s.CheckDeclaration(d.list.At(i).(*AST.Decl))
		}
		
	} else {
		// single declaration
		switch d.tok {
		case Scanner.IMPORT:
		case Scanner.EXPORT:
		case Scanner.CONST:
		case Scanner.VAR:
		case Scanner.TYPE:
		case Scanner.FUNC:
		default:
			unreachable();
		}
	}
}


func (s *State) CheckProgram(p *AST.Program) {
	for i := 0; i < p.decls.Len(); i++ {
		s.CheckDeclaration(p.decls.At(i).(*AST.Decl));
	}
}


// ----------------------------------------------------------------------------

export func CheckProgram(err Scanner.ErrorHandler, p *AST.Program) {
	var s State;
	s.Init(err);
	s.CheckProgram(p);
}
