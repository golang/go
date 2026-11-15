// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

type LogosPlatformRet struct {
	Value, Slot Type
}

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
