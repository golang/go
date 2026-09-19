// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"slices"
	"strings"

	"cmd/compile/internal/types"
)

// A TypeSet is the set of dynamic types an interface-typed value may
// hold, as computed and consumed by the devirtualize package.
//
// Members are concrete types, TypeSetNil, or TypeSetUnknown. A set
// containing TypeSetUnknown holds nothing else, since unknown
// subsumes every other member; its presence can always be checked by
// ts[0] == TypeSetUnknown. TypeSet(nil) records that no value has
// been seen, which consumers must treat as unknown, not as proof
// that no value exists.
//
// The type lives here rather than in devirtualize so that [Func] can
// hold one, like it holds an [Inline].
type TypeSet []*types.Type

// Sentinel members: the nil interface, treated as a dynamic type of
// its own, and the unknown dynamic type.
var (
	typeSetNil, typeSetUnknown types.Type

	TypeSetNil     = &typeSetNil
	TypeSetUnknown = &typeSetUnknown
)

// typeSetLimit is the largest set tracked for one value; a set that
// would grow past it collapses to unknown.
const typeSetLimit = 4

// Add returns the set extended by the abstract dynamic type t.
func (ts TypeSet) Add(t *types.Type) TypeSet {
	if len(ts) == 1 && ts[0] == TypeSetUnknown {
		return ts
	}

	// Unknown subsumes all types. Also, a shaped type is one which is dependent
	// on a type dictionary (e.g. go.shape.uint64 within a generic stencil), so
	// we must treat it as unknown.
	if t == TypeSetUnknown || t.HasShape() {
		return TypeSet{TypeSetUnknown}
	}

	// Pointer identity can treat two occurrences of the same unnamed
	// composite type as distinct members, since those need not share
	// a *types.Type. Types with methods are named or pointers to
	// named types and are represented uniquely, so for the types this
	// analysis cares about, pointer identity is exact.
	if slices.Contains(ts, t) {
		return ts
	}
	if len(ts) >= typeSetLimit {
		return TypeSet{TypeSetUnknown}
	}
	return append(ts, t)
}

// String formats the set for debug output and diagnostics.
func (ts TypeSet) String() string {
	names := make([]string, len(ts))
	for i, t := range ts {
		names[i] = typeSetName(t)
	}
	return strings.Join(names, ", ")
}

// typeSetName formats one abstract dynamic type for diagnostics.
func typeSetName(t *types.Type) string {
	switch t {
	case TypeSetNil:
		return "<nil>"
	case TypeSetUnknown:
		return "<unknown>"
	}
	return t.String()
}
