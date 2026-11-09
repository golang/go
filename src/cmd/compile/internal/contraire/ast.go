// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package contraire

type prog struct{ decls []decl }

// ============================== Decls ===============================

type decl interface{ decl() }

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

type constExpr interface{ constExpr() }

type constName struct{ name string }
type constLtrl struct{}

func (*constName) constExpr() {}
func (*constLtrl) constExpr() {}

// ============================== Types ===============================

type typeExpr interface{ typeExpr() }

type typeName struct{ name string }
type typeBasic struct{}
type typePointer struct{ expr typeExpr }

func (*typeName) typeExpr()    {}
func (*typeBasic) typeExpr()   {}
func (*typePointer) typeExpr() {}
