// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.23

package cursor

import (
	"go/ast"
	_ "unsafe" // for go:linkname

	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/astutil/edge"
)

// This file defines backdoor access to inspector.

// Copied from inspector.event; must remain in sync.
// (Note that the linkname effects a type coercion too.)
type event struct {
	node   ast.Node
	typ    uint64 // typeOf(node) on push event, or union of typ strictly between push and pop events on pop events
	index  int32  // index of corresponding push or pop event (relative to this event's index, +ve=push, -ve=pop)
	parent int32  // index of parent's push node (push nodes only); or edge and index, bit packed (pop nodes only)
}

//go:linkname maskOf golang.org/x/tools/go/ast/inspector.maskOf
func maskOf(nodes []ast.Node) uint64

//go:linkname events golang.org/x/tools/go/ast/inspector.events
func events(in *inspector.Inspector) []event

//go:linkname packEdgeKindAndIndex golang.org/x/tools/go/ast/inspector.packEdgeKindAndIndex
func packEdgeKindAndIndex(edge.Kind, int) int32

//go:linkname unpackEdgeKindAndIndex golang.org/x/tools/go/ast/inspector.unpackEdgeKindAndIndex
func unpackEdgeKindAndIndex(int32) (edge.Kind, int)

func (c Cursor) events() []event { return events(c.in) }
