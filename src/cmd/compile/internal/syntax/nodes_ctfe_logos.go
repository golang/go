// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M5-S2 (`const func`
// declarations; SPEC Sec 2.1, T-201). Upstream files reference it
// through hook lines only (G3); the rule below runs solely in LogosMode,
// so plain Go parses exactly as upstream.

// ConstFuncDecl is a loGo/s `const func` declaration (SPEC T-201).
//
// Func is the declared function, parsed exactly as a plain FuncDecl
// (receivers, type parameters, and bodies included); ConstPos is the
// position of `const`. The type checker records constness, checks the
// purity rules, and evaluates calls in const initializers; lowering
// erases the marker by splicing `const` away, so later passes only ever
// see the inner FuncDecl.
type ConstFuncDecl struct {
	ConstPos Pos // position of `const`
	Func     *FuncDecl
	decl
}
