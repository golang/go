// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// LoGo/s M5-S5 type rules for zero-copy conversions (SPEC Sec 2.1,
// T-201). `string` converts to `const []byte` without copying, and a
// `const []byte` converts back to `string` without copying only when
// its provenance is a string or a const composite; any other const
// view converts with a copy. Provenance itself is tracked by the
// lowering (logos/zerocopy.go), which sees every creation site and
// every local flow; this file only admits the conversions the
// lowering can make zero-copy.
//
// Upstream files call in through hook lines only: call.go (target-view
// conversions). M5-S1's logosConstConvert keeps owning the reverse
// direction (conversions whose source is a const view), so the two
// hooks never overlap: this one requires a ConstView target, the other
// a ConstView source. The parser only produces a ConstType conversion
// function in LogosMode, so this hook is unreachable in plain Go.

// logosConstTypeExpr checks a `const T` type in expression position
// (SPEC T-201): a conversion function such as `const []byte` in
// `const []byte(s)` reaches the checker as a bare ConstType operand,
// which the upstream expression switch does not route. It takes the
// same type as the M5-S1 declaration rule and reports a type
// expression, so the caller's conversion path runs unchanged.
func (check *Checker) logosConstTypeExpr(x *operand, e *syntax.ConstType) exprKind {
	x.mode_ = typexpr
	x.typ_ = check.logosConstType(e, nil)
	x.expr = e
	return expression
}

// logosZeroCopyConvert checks a conversion whose target type is a
// const view (SPEC T-201): the source must convert to the viewed
// element type under the ordinary conversion rules, and the result
// takes the view type. A view converts to a view with an identical
// element directly. It reports true when it handled the conversion
// (even on failure, when the element conversion already reported the
// error); otherwise it reports false and the caller runs the upstream
// path, whose error names the view type it cannot convert to.
func (check *Checker) logosZeroCopyConvert(x *operand, T Type) bool {
	if !check.conf.Logos || !x.isValid() {
		return false
	}
	cdst := logosAsConstView(T)
	if cdst == nil {
		return false
	}
	if csrc := logosAsConstView(x.typ()); csrc != nil {
		if !Identical(csrc.Elem, cdst.Elem) {
			return false
		}
		x.typ_ = T
		return true
	}
	check.conversion(x, cdst.Elem)
	if !x.isValid() {
		return true
	}
	if x.mode() == constant_ {
		x.mode_ = value
		x.val = nil
	}
	x.typ_ = T
	return true
}
