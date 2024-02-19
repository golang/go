// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

import (
	"fmt"
	"runtime"
	"strings"
)

// fmtFrames formats a backtrace for reporting reader/writer desyncs.
func fmtFrames(pcs ...uintptr) []string {
	res := make([]string, 0, len(pcs))
	walkFrames(pcs, func(file string, line int, name string, offset uintptr) {
		// Trim package from function name. It's just redundant noise.
		name = strings.TrimPrefix(name, "cmd/compile/internal/noder.")

		res = append(res, fmt.Sprintf("%s:%v: %s +0x%v", file, line, name, offset))
	})
	return res
}

type frameVisitor func(file string, line int, name string, offset uintptr)

// walkFrames calls visit for each call frame represented by pcs.
//
// pcs should be a slice of PCs, as returned by runtime.Callers.
func walkFrames(pcs []uintptr, visit frameVisitor) {
	if len(pcs) == 0 {
		return
	}

	frames := runtime.CallersFrames(pcs)
	for {
		frame, more := frames.Next()
		visit(frame.File, frame.Line, frame.Function, frame.PC-frame.Entry)
		if !more {
			return
		}
	}
}

// SyncMarker is an enum type that represents markers that may be
// written to export data to ensure the reader and writer stay
// synchronized.
type SyncMarker int

//go:generate stringer -type=SyncMarker -trimprefix=Sync

const (
	_ SyncMarker = iota

	// Public markers (known to go/types importers).

	// Low-level coding markers.
	SyncEOF
	SyncBool
	SyncInt64
	SyncUint64
	SyncString
	SyncValue
	SyncVal
	SyncRelocs
	SyncReloc
	SyncUseReloc

	// Higher-level object and type markers.
	SyncPublic
	SyncPos
	SyncPosBase
	SyncObject
	SyncObject1
	SyncPkg
	SyncPkgDef
	SyncMethod
	SyncType
	SyncTypeIdx
	SyncTypeParamNames
	SyncSignature
	SyncParams
	SyncParam
	SyncCodeObj
	SyncSym
	SyncLocalIdent
	SyncSelector

	// Private markers (only known to cmd/compile).
	SyncPrivate

	SyncFuncExt
	SyncVarExt
	SyncTypeExt
	SyncPragma

	SyncExprList
	SyncExprs
	SyncExpr
	SyncExprType
	SyncAssign
	SyncOp
	SyncFuncLit
	SyncCompLit

	SyncDecl
	SyncFuncBody
	SyncOpenScope
	SyncCloseScope
	SyncCloseAnotherScope
	SyncDeclNames
	SyncDeclName

	SyncStmts
	SyncBlockStmt
	SyncIfStmt
	SyncForStmt
	SyncFourStmt
	SyncUnlessStmt
	SyncSwitchStmt
	SyncRangeStmt
	SyncCaseClause
	SyncCommClause
	SyncSelectStmt
	SyncDecls
	SyncLabeledStmt
	SyncUseObjLocal
	SyncAddLocal
	SyncLinkname
	SyncStmt1
	SyncStmtsEnd
	SyncLabel
	SyncOptLabel

	SyncMultiExpr
	SyncRType
	SyncConvRTTI
)
