// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import "internal/pkgbits"

type codeStmt int

func (c codeStmt) Marker() pkgbits.SyncMarker { return pkgbits.SyncStmt1 }
func (c codeStmt) Value() int                 { return int(c) }

const (
	stmtEnd codeStmt = iota
	stmtLabel
	stmtBlock
	stmtExpr
	stmtSend
	stmtAssign
	stmtAssignOp
	stmtIncDec
	stmtBranch
	stmtCall
	stmtReturn
	stmtIf
	stmtFor
	stmtSwitch
	stmtSelect
)

type codeExpr int

func (c codeExpr) Marker() pkgbits.SyncMarker { return pkgbits.SyncExpr }
func (c codeExpr) Value() int                 { return int(c) }

// TODO(mdempsky): Split expr into addr, for lvalues.
const (
	exprConst  codeExpr = iota
	exprLocal           // local variable
	exprGlobal          // global variable or function
	exprCompLit
	exprFuncLit
	exprSelector
	exprIndex
	exprSlice
	exprAssert
	exprUnaryOp
	exprBinaryOp
	exprCall
	exprConvert
	exprNew
	exprMake
)

type codeAssign int

func (c codeAssign) Marker() pkgbits.SyncMarker { return pkgbits.SyncAssign }
func (c codeAssign) Value() int                 { return int(c) }

const (
	assignBlank codeAssign = iota
	assignDef
	assignExpr
)

type codeDecl int

func (c codeDecl) Marker() pkgbits.SyncMarker { return pkgbits.SyncDecl }
func (c codeDecl) Value() int                 { return int(c) }

const (
	declEnd codeDecl = iota
	declFunc
	declMethod
	declVar
	declOther
)
