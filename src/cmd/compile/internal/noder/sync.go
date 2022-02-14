// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"strings"
)

// enableSync controls whether sync markers are written into unified
// IR's export data format and also whether they're expected when
// reading them back in. They're inessential to the correct
// functioning of unified IR, but are helpful during development to
// detect mistakes.
//
// When sync is enabled, writer stack frames will also be included in
// the export data. Currently, a fixed number of frames are included,
// controlled by -d=syncframes (default 0).
const enableSync = true

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

// syncMarker is an enum type that represents markers that may be
// written to export data to ensure the reader and writer stay
// synchronized.
type syncMarker int

//go:generate stringer -type=syncMarker -trimprefix=sync

const (
	_ syncMarker = iota

	// Public markers (known to go/types importers).

	// Low-level coding markers.

	syncEOF
	syncBool
	syncInt64
	syncUint64
	syncString
	syncValue
	syncVal
	syncRelocs
	syncReloc
	syncUseReloc

	// Higher-level object and type markers.
	syncPublic
	syncPos
	syncPosBase
	syncObject
	syncObject1
	syncPkg
	syncPkgDef
	syncMethod
	syncType
	syncTypeIdx
	syncTypeParamNames
	syncSignature
	syncParams
	syncParam
	syncCodeObj
	syncSym
	syncLocalIdent
	syncSelector

	// Private markers (only known to cmd/compile).
	syncPrivate

	syncFuncExt
	syncVarExt
	syncTypeExt
	syncPragma

	syncExprList
	syncExprs
	syncExpr
	syncOp
	syncFuncLit
	syncCompLit

	syncDecl
	syncFuncBody
	syncOpenScope
	syncCloseScope
	syncCloseAnotherScope
	syncDeclNames
	syncDeclName

	syncStmts
	syncBlockStmt
	syncIfStmt
	syncForStmt
	syncSwitchStmt
	syncRangeStmt
	syncCaseClause
	syncCommClause
	syncSelectStmt
	syncDecls
	syncLabeledStmt
	syncUseObjLocal
	syncAddLocal
	syncLinkname
	syncStmt1
	syncStmtsEnd
	syncLabel
	syncOptLabel
)
