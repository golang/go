// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types2"
)

// M6-S3 soft diagnostics (SPEC Sec 1.2): warn-severity loGo/s rules
// report through the build without failing it. The predicate matches
// types2's firstErr exemption exactly, so upstream soft errors
// (duplicate labels, invalid type arguments) keep failing through
// ErrorfAt.

// logosReportSoftWarn reports a warn-severity loGo/s diagnostic as a
// warning and reports true. Anything else returns false for the
// upstream path.
func logosReportSoftWarn(m *posMap, terr types2.Error) bool {
	if !types2.LogosWarnOnly(terr) {
		return false
	}
	base.WarnfAt(m.makeXPos(terr.Pos), "%s", terr.Msg)
	return true
}
