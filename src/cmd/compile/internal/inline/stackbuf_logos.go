// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

// logosUpstreamVariableMakeThreshold is the upstream stack-buffer size in
// bytes for variable-size make and append backing stores
// (base.Debug.VariableMakeThreshold before any -O tier sets it).
const logosUpstreamVariableMakeThreshold = 32

// LogosVariableMakeThreshold returns the stack-buffer size in bytes that
// walk (variable-size make) and ssagen (append backing stores) use for the
// sites that land in fn. The slice pass reads the tier threshold directly
// and needs no hook: it only picks candidates, and a candidate that
// ssagen then gives no buffer lowers its move2heap to a no-op.
//
// Each such site takes its own [K]E buffer of that size in the frame of the
// function that holds the site, inlined bodies included, and the buffers
// are not shared. A recursive function pays its frame once per recursion
// level, so a tier threshold of 4096 turns fifteen inlined sites into a
// 60 KB frame and a 10,000-level recursion into a stack overflow
// (encoding/asn1 parseField at -O=3). A function in a recursive call-graph
// cycle therefore keeps the upstream threshold whatever the tier asks for,
// and every other function gets the tier threshold. A threshold set below
// the upstream value stays as set.
func LogosVariableMakeThreshold(fn *ir.Func) int64 {
	t := int64(base.Debug.VariableMakeThreshold)
	if t > logosUpstreamVariableMakeThreshold && logosRecursiveMarks[fn] {
		return logosUpstreamVariableMakeThreshold
	}
	return t
}
