// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

var LogosLockFacts func(pkg *types2.Package, info *types2.Info, files []*syntax.File) []LockFact

func writeLocks(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *Facts) {
	if LogosLockFacts == nil {
		return
	}
	for _, l := range LogosLockFacts(pkg, info, files) {
		f.Locks = appendUnique(f.Locks, l)
	}
}
