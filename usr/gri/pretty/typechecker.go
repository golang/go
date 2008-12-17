// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package TypeChecker

import (
	AST "ast";
	Universe "universe";
	Globals "globals";
	Object "object";
	Type "type";
)


type State struct {
	level int;
	top_scope *Globals.Scope;
}


// ----------------------------------------------------------------------------
// Support

func (s *State) Error(pos int, msg string) {
	panicln("error:" + msg);
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

func (s *State) DeclareIdent(kind int) {
	obj := Globals.NewObject(0, kind, "");
	s.Declare(obj);
}


// ----------------------------------------------------------------------------

func (s *State) CheckProgram(p *AST.Program) {
	s.OpenScope();
	
	{	s.OpenScope();
	
		s.CloseScope();
	}
	
	s.CloseScope();
}


// ----------------------------------------------------------------------------

export func CheckProgram(p *AST.Program) {
	var s State;
	s.CheckProgram(p);
}
