// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// LogosPlatformRet records one return operand that holds a platform pointer
// of type Value and converts to the interface result type Slot.
type LogosPlatformRet struct {
	Value, Slot Type
}

// logosPlatformReturn records a return operand that holds a platform
// pointer (SPEC Sec 2.2, T-202) and converts to an interface result. The
// lint rule correctness::typed_nil_return (SPEC Sec 1.2, M9-S2) reads
// Info.LogosPlatformRets, because provenance is operand derivation, not a
// type, and only the checker can answer it. A nil pointer boxed into an
// interface makes a non-nil interface, so the lint rule asks for a nil
// check. The record is keyed by the start position of the operand, with the
// operand type and the result type: the results of one call that fills a
// whole result list are synthetic operands that share the start of the call,
// and the two types tell them apart. It never reports an error. Hooked as a
// bare call from assignment.
func (check *Checker) logosPlatformReturn(x *operand, T Type, context string) {
	if x == nil || !check.isLogosFile(x) || context != "return statement" || check.Info == nil {
		return
	}
	if T == nil || !isNonTypeParamInterface(T) || !x.isValid() || x.expr == nil || x.isNil() {
		return
	}
	if !check.logosIsPlatform(x) {
		return
	}
	if check.Info.LogosPlatformRets == nil {
		check.Info.LogosPlatformRets = make(map[syntax.Pos][]LogosPlatformRet)
	}
	pos := syntax.StartPos(x.expr)
	check.Info.LogosPlatformRets[pos] = append(check.Info.LogosPlatformRets[pos], LogosPlatformRet{Value: x.typ(), Slot: T})
}
