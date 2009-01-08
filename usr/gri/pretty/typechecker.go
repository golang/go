// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package TypeChecker

import (
	AST "ast";
	Scanner "scanner";
	Universe "universe";
	Globals "globals";
	Object "object";
	Type "type";
)


type State struct {
	// setup
	err Scanner.ErrorHandler;

	// state
	level int;
	top_scope *Globals.Scope;
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
// Scopes

func (s *State) OpenScope() {
	s.top_scope = Globals.NewScope(s.top_scope);
}


func (s *State) CloseScope() {
	s.top_scope = s.top_scope.parent;
}


func (s *State) Lookup(ident string) *Globals.Object {
	for scope := s.top_scope; scope != nil; scope = scope.parent {
		obj := scope.Lookup(ident);
		if obj != nil {
			return obj;
		}
	}
	return nil;
}


func (s *State) DeclareInScope(scope *Globals.Scope, obj *Globals.Object) {
	if s.level > 0 {
		panic("cannot declare objects in other packages");
	}
	obj.pnolev = s.level;
	if scope.Lookup(obj.ident) != nil {
		s.Error(obj.pos, `"` + obj.ident + `" is declared already`);
		return;  // don't insert it into the scope
	}
	scope.Insert(obj);
}


func (s *State) Declare(obj *Globals.Object) {
	s.DeclareInScope(s.top_scope, obj);
}


// ----------------------------------------------------------------------------
// Common productions

func (s *State) DeclareIdent(ident *AST.Expr, kind int, typ *AST.Type) {
	// ident is either a comma-separated list or a single ident
	switch ident.tok {
	case Scanner.IDENT:
		obj := Globals.NewObject(ident.pos, kind, ident.obj.ident);
		s.Declare(obj);
	case Scanner.COMMA:
		s.DeclareIdent(ident.x, kind, typ);
		s.DeclareIdent(ident.y, kind, typ);		
	default:
		unreachable();
	}
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
			assert(d.ident == nil || d.ident.tok == Scanner.IDENT);
			if d.ident != nil {
				s.DeclareIdent(d.ident, d.tok, d.typ);
			} else {
			}

		case Scanner.EXPORT:
			// TODO

		case Scanner.CONST:
			s.DeclareIdent(d.ident, d.tok, d.typ);

		case Scanner.VAR:
			s.DeclareIdent(d.ident, d.tok, d.typ);

		case Scanner.TYPE:
			assert(d.ident.tok == Scanner.IDENT);
			// types may be forward-declared
			obj := s.Lookup(d.ident.obj.ident);
			if obj != nil {
				// TODO check if proper forward-declaration

			} else {
				s.DeclareIdent(d.ident, d.tok, d.typ);
			}

		case Scanner.FUNC:
			assert(d.ident.tok == Scanner.IDENT);
			if d.typ.key != nil {
				// method
				// TODO
			} else {
				// functions may be forward-declared
				obj := s.Lookup(d.ident.obj.ident);
				if obj != nil {
				  // TODO check if proper forward-declaration
				  
				} else {
					s.DeclareIdent(d.ident, d.tok, d.typ);
				}
			}

		default:
			unreachable();
		}
	}
}


func (s *State) CheckProgram(p *AST.Program) {
	s.OpenScope();
	
	{	s.OpenScope();
		for i := 0; i < p.decls.Len(); i++ {
			s.CheckDeclaration(p.decls.At(i).(*AST.Decl));
		}
		s.CloseScope();
	}
	
	s.CloseScope();
}


// ----------------------------------------------------------------------------

export func CheckProgram(err Scanner.ErrorHandler, p *AST.Program) {
	var s State;
	s.Init(err);
	s.CheckProgram(p);
}
