// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed package import.
// See iexport.go for the export data format.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// HaveInlineBody reports whether we have fn's inline body available
// for inlining.
//
// It's a function literal so that it can be overridden for
// GOEXPERIMENT=unified.
var HaveInlineBody = func(fn *ir.Func) bool {
	base.Fatalf("HaveInlineBody not overridden")
	panic("unreachable")
}

func SetBaseTypeIndex(t *types.Type, i, pi int64) {
	if t.Obj() == nil {
		base.Fatalf("SetBaseTypeIndex on non-defined type %v", t)
	}
	if i != -1 && pi != -1 {
		typeSymIdx[t] = [2]int64{i, pi}
	}
}

// Map imported type T to the index of type descriptor symbols of T and *T,
// so we can use index to reference the symbol.
// TODO(mdempsky): Store this information directly in the Type's Name.
var typeSymIdx = make(map[*types.Type][2]int64)

func BaseTypeIndex(t *types.Type) int64 {
	tbase := t
	if t.IsPtr() && t.Sym() == nil && t.Elem().Sym() != nil {
		tbase = t.Elem()
	}
	i, ok := typeSymIdx[tbase]
	if !ok {
		return -1
	}
	if t != tbase {
		return i[1]
	}
	return i[0]
}
