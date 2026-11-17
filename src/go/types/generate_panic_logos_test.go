// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import "bytes"

// goTypesPanicSource strips the LoGOs panic-recording hook when generating
// go/types. Compiler types2 calls check.logosRecordVoidPanic to record
// resultless panic calls for LoGOs option narrowing; public go/types has
// neither that mode nor the hook. The exact source shape must occur once so a
// future checker edit fails closed.
func goTypesPanicSource(src []byte) []byte {
	const logos = `	case _Panic:
		// panic(x)
		// record panic call if inside a function with result parameters
		// (for use in Checker.isTerminating)
		if check.sig != nil && check.sig.results.Len() > 0 {
			// function has result parameters
			p := check.isPanic
			if p == nil {
				// allocate lazily
				p = make(map[*syntax.CallExpr]bool)
				check.isPanic = p
			}
			p[call] = true
		}
		check.logosRecordVoidPanic(call)
`
	const upstream = `	case _Panic:
		// panic(x)
		// record panic call if inside a function with result parameters
		// (for use in Checker.isTerminating)
		if check.sig != nil && check.sig.results.Len() > 0 {
			// function has result parameters
			p := check.isPanic
			if p == nil {
				// allocate lazily
				p = make(map[*syntax.CallExpr]bool)
				check.isPanic = p
			}
			p[call] = true
		}
`
	if bytes.Count(src, []byte(logos)) != 1 {
		panic("go/types generation: expected exactly one loGo/s panic registration guard")
	}
	return bytes.Replace(src, []byte(logos), []byte(upstream), 1)
}
