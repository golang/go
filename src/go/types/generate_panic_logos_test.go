// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import "bytes"

// goTypesPanicSource restores the upstream panic-registration rule when
// generating go/types. Compiler types2 also records resultless panic calls for
// loGo/s option narrowing; public go/types has neither that mode nor its flag.
// The exact source shape must occur once so a future checker edit fails closed.
func goTypesPanicSource(src []byte) []byte {
	const logos = `	case _Panic:
		// panic(x)
		// Record panic calls for result-bearing function termination and
		// for loGo/s option narrowing, including functions without results.
		if check.sig != nil && (check.sig.results.Len() > 0 || check.conf.Logos) {
`
	const upstream = `	case _Panic:
		// panic(x)
		// record panic call if inside a function with result parameters
		// (for use in Checker.isTerminating)
		if check.sig != nil && check.sig.results.Len() > 0 {
			// function has result parameters
`
	if bytes.Count(src, []byte(logos)) != 1 {
		panic("go/types generation: expected exactly one loGo/s panic registration guard")
	}
	return bytes.Replace(src, []byte(logos), []byte(upstream), 1)
}
