// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/logos"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
)

func logosGoCallerHook(pkg *types2.Package, info *types2.Info, files []*syntax.File) bool {
	if !logos.LowerGoCalls(pkg, info, files) {
		return false
	}
	for _, p := range types.PkgMap() {
		p.Direct = false
	}
	typecheck.Target.Imports = nil
	return true
}
