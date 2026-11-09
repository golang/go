// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package contraire

type prog struct{ decls []decl }

// ============================== Decls ===============================

type decl interface {
	node
	decl()
}

type constDecl struct {
	name string
	expr constExpr
}

type typeDecl struct {
	name string
	expr typeExpr
}

func (*constDecl) decl() {}
func (*typeDecl) decl()  {}

// ============================== Consts ==============================

type constExpr interface {
	node
	constExpr()
}

type constName struct{ name string }
type constLtrl struct{}

func (*constName) constExpr() {}
func (*constLtrl) constExpr() {}

// ============================== Types ===============================

type typeExpr interface {
	node
	typeExpr()
}

type typeName struct{ name string }
type typeBasic struct{}
type typePointer struct{ expr typeExpr }

func (*typeName) typeExpr()    {}
func (*typeBasic) typeExpr()   {}
func (*typePointer) typeExpr() {}

// ============================== Visitor =============================

type node interface {
	receive(visitor)
}

type visitor interface {
	visitProg(*prog)
	visitConstDecl(*constDecl)
	visitTypeDecl(*typeDecl)
	visitConstName(*constName)
	visitConstLtrl(*constLtrl)
	visitTypeName(*typeName)
	visitTypeBasic(*typeBasic)
	visitTypePointer(*typePointer)
}

func (x *prog) receive(v visitor)        { v.visitProg(x) }
func (x *constDecl) receive(v visitor)   { v.visitConstDecl(x) }
func (x *typeDecl) receive(v visitor)    { v.visitTypeDecl(x) }
func (x *constName) receive(v visitor)   { v.visitConstName(x) }
func (x *constLtrl) receive(v visitor)   { v.visitConstLtrl(x) }
func (x *typeName) receive(v visitor)    { v.visitTypeName(x) }
func (x *typeBasic) receive(v visitor)   { v.visitTypeBasic(x) }
func (x *typePointer) receive(v visitor) { v.visitTypePointer(x) }
