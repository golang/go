// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"strings"
)

func panicOnCall(f *Func) {
	if len(base.PanicOnCallPatterns) == 0 {
		return
	}
	if skipPanicOnCallPackage(base.Ctxt.Pkgpath) {
		return
	}

	matcher := newSymbolMatcher(base.PanicOnCallPatterns)
	for _, b := range f.Blocks {
		for i := 0; i < len(b.Values); i++ {
			v := b.Values[i]
			if v.Op != OpStaticCall && v.Op != OpStaticLECall {
				continue
			}
			aux, ok := v.Aux.(*AuxCall)
			if !ok || aux == nil || aux.Fn == nil {
				continue
			}
			target := aux.Fn.Name
			if !matcher.match(target) {
				continue
			}
			memIdx := len(v.Args) - 1
			if memIdx < 0 {
				continue
			}
			mem := v.Args[memIdx]

			// Build args without leaving OpConstString to avoid lowerer ICEs:
			// take address of string data and length, pass as (*byte, uintptr).
			_, sb := f.spSb()
			dataSym := f.fe.StringData(target)
			ptr := b.NewValue1A(v.Pos, OpAddr, f.Config.Types.BytePtr, symToAux(dataSym), sb)
			lenVal := b.NewValue0I(v.Pos, OpConst64, f.Config.Types.Uintptr, int64(len(target)))

			auxCall := StaticAuxCall(ir.Syms.PanicOnCall, f.ABIDefault.ABIAnalyzeTypes([]*types.Type{
				f.Config.Types.BytePtr,      // data pointer
				types.Types[types.TUINTPTR], // length
			}, nil))
			panicCall := b.NewValue0A(v.Pos, OpStaticCall, types.TypeMem, auxCall)
			panicCall.AuxInt = auxCall.ABIInfo().ArgWidth()
			panicCall.AddArg(ptr)
			panicCall.AddArg(lenVal)
			panicCall.AddArg(mem)

			// Insert ptr, len, panicCall before the call to keep scheduling sane.
			// Note: b.NewValue... already appended these 3 to b.Values, so exclude them.
			vals := b.Values
			newVals := make([]*Value, 0, len(vals))
			newVals = append(newVals, vals[:i]...)
			newVals = append(newVals, ptr, lenVal, panicCall)
			newVals = append(newVals, vals[i:len(vals)-3]...) // exclude the 3 auto-appended values
			b.Values = newVals

			// Wire memory so panic runs before the target call.
			v.SetArg(memIdx, panicCall)
			i += 3 // skip over inserted values
		}
	}
}

func skipPanicOnCallPackage(pkgPath string) bool {
	if pkgPath == "" {
		return true
	}
	if strings.HasPrefix(pkgPath, "cmd/") ||
		strings.HasPrefix(pkgPath, "runtime") ||
		strings.HasPrefix(pkgPath, "internal/") ||
		strings.HasPrefix(pkgPath, "bootstrap") {
		return true
	}
	if !strings.Contains(pkgPath, ".") &&
		pkgPath != "main" &&
		pkgPath != "command-line-arguments" &&
		!strings.HasSuffix(pkgPath, "_test") {
		return true
	}
	return false
}

type symbolMatcher struct {
	exact   map[string]struct{}
	prefix  []string
	enabled bool
}

func newSymbolMatcher(patterns []string) symbolMatcher {
	m := symbolMatcher{
		exact: make(map[string]struct{}, len(patterns)),
	}
	for _, p := range patterns {
		if strings.HasSuffix(p, "*") {
			m.prefix = append(m.prefix, strings.TrimSuffix(p, "*"))
			continue
		}
		m.exact[p] = struct{}{}
	}
	m.enabled = len(m.exact) > 0 || len(m.prefix) > 0
	return m
}

func (m symbolMatcher) match(sym string) bool {
	if !m.enabled {
		return false
	}
	if _, ok := m.exact[sym]; ok {
		return true
	}
	for _, pre := range m.prefix {
		if strings.HasPrefix(sym, pre) {
			return true
		}
	}
	return false
}
