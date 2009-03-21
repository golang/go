// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package TypeChecker

import (
	"token";
	"scanner";
	"ast";
)


type state struct {
	// setup
	err scanner.ErrorHandler;
}


func (s *state) Init(err scanner.ErrorHandler) {
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


func (s *state) Error(loc scanner.Location, msg string) {
	s.err.Error(loc, msg);
}


// ----------------------------------------------------------------------------

func (s *state) CheckType() {
}


/*
func (s *state) CheckDeclaration(d *AST.Decl) {
	if d.Tok != token.FUNC && d.List != nil {
		// group of parenthesized declarations
		for i := 0; i < d.List.Len(); i++ {
			s.CheckDeclaration(d.List.At(i).(*AST.Decl))
		}

	} else {
		// single declaration
		switch d.Tok {
		case token.IMPORT:
		case token.CONST:
		case token.VAR:
		case token.TYPE:
		case token.FUNC:
		default:
			unreachable();
		}
	}
}
*/


func (s *state) CheckProgram(p *ast.Package) {
	for i := 0; i < len(p.Decls); i++ {
		//s.CheckDeclaration(p.Decls[i].(*AST.Decl));
	}
}


// ----------------------------------------------------------------------------

func CheckProgram(err scanner.ErrorHandler, p *ast.Package) {
	var s state;
	s.Init(err);
	s.CheckProgram(p);
}
