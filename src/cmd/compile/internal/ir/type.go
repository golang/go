// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func TypeNode(t *types.Type) Node {
	return TypeNodeAt(src.NoXPos, t)
}

func TypeNodeAt(pos src.XPos, t *types.Type) Node {
	// if we copied another type with *t = *u
	// then t->nod might be out of date, so
	// check t->nod->type too
	if AsNode(t.Nod) == nil || AsNode(t.Nod).Type() != t {
		t.Nod = NodAt(pos, OTYPE, nil, nil)
		AsNode(t.Nod).SetType(t)
		AsNode(t.Nod).SetSym(t.Sym)
	}

	return AsNode(t.Nod)
}
