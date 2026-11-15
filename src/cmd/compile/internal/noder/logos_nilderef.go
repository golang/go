// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func logosReportNilDeref(m *posMap, files []*syntax.File, info *types2.Info, terr types2.Error) bool {
	if !strings.HasPrefix(terr.Msg, lint.NilDerefTag) {
		return false
	}
	f := logosFileAt(files, terr.Pos)
	if f == nil {
		return false
	}
	stashed, _ := nextInputs([]*syntax.File{f})
	src, ok := stashed[f]
	if !ok {
		return false
	}
	r, ok := lint.NilDerefResult(info, f, src, terr.Pos, terr.Msg)
	if !ok {
		return false
	}
	base.LogosReport(logosDiag(m, r))
	return true
}

func logosFileAt(files []*syntax.File, pos syntax.Pos) *syntax.File {
	for _, f := range files {
		if f.Pos().FileBase() == pos.FileBase() {
			return f
		}
	}
	return nil
}
