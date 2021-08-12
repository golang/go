// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

type code interface {
	marker() syncMarker
	value() int
}

type codeVal int

func (c codeVal) marker() syncMarker { return syncVal }
func (c codeVal) value() int         { return int(c) }

const (
	valBool codeVal = iota
	valString
	valInt64
	valBigInt
	valBigRat
	valBigFloat
)

type codeType int

func (c codeType) marker() syncMarker { return syncType }
func (c codeType) value() int         { return int(c) }

const (
	typeBasic codeType = iota
	typeNamed
	typePointer
	typeSlice
	typeArray
	typeChan
	typeMap
	typeSignature
	typeStruct
	typeInterface
	typeUnion
	typeTypeParam
)

type codeObj int

func (c codeObj) marker() syncMarker { return syncCodeObj }
func (c codeObj) value() int         { return int(c) }

const (
	objAlias codeObj = iota
	objConst
	objType
	objFunc
	objVar
	objStub
)

type codeStmt int

func (c codeStmt) marker() syncMarker { return syncStmt1 }
func (c codeStmt) value() int         { return int(c) }

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

	// TODO(mdempsky): Remove after we don't care about toolstash -cmp.
	stmtTypeDeclHack
)

type codeExpr int

func (c codeExpr) marker() syncMarker { return syncExpr }
func (c codeExpr) value() int         { return int(c) }

// TODO(mdempsky): Split expr into addr, for lvalues.
const (
	exprNone codeExpr = iota
	exprConst
	exprType  // type expression
	exprLocal // local variable
	exprName  // global variable or function
	exprBlank
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
)

type codeDecl int

func (c codeDecl) marker() syncMarker { return syncDecl }
func (c codeDecl) value() int         { return int(c) }

const (
	declEnd codeDecl = iota
	declFunc
	declMethod
	declVar
	declOther
)
