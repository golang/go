// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
)

// LoGo/s M7-S4 single-parameter substitution for `const switch`
// (SPEC Sec 2.5, T-205). The rebinding of an exact case replaces the
// switched type parameter with the case type in the type of each
// case-scope variable.

// logosSubstOne returns typ with tpar replaced by repl, reusing the
// instantiation substitution machinery.
func logosSubstOne(check *Checker, pos syntax.Pos, typ Type, tpar *TypeParam, repl Type) Type {
	return check.subst(pos, typ, makeSubstMap([]*TypeParam{tpar}, []Type{repl}), nil, check.context())
}
