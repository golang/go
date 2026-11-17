// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

const logosUpstreamVariableMakeThreshold = 32

func LogosVariableMakeThreshold(fn *ir.Func) int64 {
	t := int64(base.Debug.VariableMakeThreshold)
	if t > logosUpstreamVariableMakeThreshold && (logosRecursiveMarks[fn] || (fn != nil && fn.Pragma&ir.Nosplit != 0)) {
		// A raised threshold can reserve a multi-KB backing array inside a
		// nosplit frame, which must stay under the 792-byte StackNosplit
		// chain limit.
		return logosUpstreamVariableMakeThreshold
	}
	return t
}
