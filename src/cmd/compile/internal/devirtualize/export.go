// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// This file moves result sets and split variants across package
// boundaries.
//
// Both live on [ir.Func], recorded during the interleaved pass and
// exported when export data is finalized, the same way escape tags
// and inlining costs travel: the linker rewrites each function's
// funcExt record with an [ExportedResults], and the reader turns the
// record back into the same ResultTypeSets and DevirtVariant fields
// the pass fills for local functions. A caller in another package
// then devirtualizes through an imported function exactly as it
// would through a local one, including calling the imported f.dv
// variant, which the linker resolves to the copy compiled in the
// defining package.
//
// Set members cross packages as pkgbits object references: a member
// is a pointer chain over a named, non-instantiated type, encoded as
// its depth plus a relocation to the type's object, which also forces
// the object into the export data. A member that cannot be encoded
// this way makes the whole slot unknown rather than an incomplete
// set, since a set claims to list every possible dynamic type.
// Resolution failures on the reading side degrade to unknown the same
// way.

// An ExportedResults is a function's result sets and split variant,
// as written into its funcExt record.
type ExportedResults struct {
	Slots []ExportedSet
	Split *ExportedSplit
}

// An ExportedSet is one result slot's type set in exportable form.
type ExportedSet struct {
	Members []ExportedType
	HasNil  bool
	Unknown bool
}

// An ExportedType names one concrete set member.
type ExportedType struct {
	Depth int // pointer depth over the named type
	Sym   *types.Sym
}

// An ExportedSplit describes a function's devirtualized variant.
type ExportedSplit struct {
	// Devirt reports, per result slot, whether the variant returns
	// the slot's single concrete type unboxed. The type itself is
	// the slot's set member.
	Devirt []bool

	// ParamNotes holds the variant's parameter escape tags, in
	// receiver-then-parameters order.
	ParamNotes []string
}

// ExportFor reports fn's record for export data, or nil if fn has
// nothing recorded.
func ExportFor(fn *ir.Func) *ExportedResults {
	rts := fn.ResultTypeSets
	if rts == nil {
		return nil
	}

	er := &ExportedResults{Slots: make([]ExportedSet, len(rts))}
	for i, set := range rts {
		er.Slots[i] = exportSet(set)
	}

	if fdv := fn.DevirtVariant; fdv != nil {
		split := &ExportedSplit{Devirt: make([]bool, len(rts))}
		for i, f := range fdv.Type().Results() {
			split.Devirt[i] = !f.Type.IsInterface() && fn.Type().Results()[i].Type.IsInterface()
		}
		for _, f := range fdv.Type().Params() {
			split.ParamNotes = append(split.ParamNotes, f.Note)
		}
		er.Split = split
	}
	return er
}

// exportSet converts one type set into exportable form.
func exportSet(set typeSet) ExportedSet {
	if len(set) == 0 {
		// No returned value was seen; importers must treat that as
		// unknown, see [typeSet].
		return ExportedSet{Unknown: true}
	}

	var es ExportedSet
	for _, t := range set {
		switch t {
		case nilType:
			es.HasNil = true
			continue
		case unknownType:
			return ExportedSet{Unknown: true}
		}

		depth := 0
		base := t
		for base.IsPtr() {
			base = base.Elem()
			depth++
		}

		sym := base.Sym()
		if sym == nil || sym.Pkg == nil || sym.Pkg.Path == "" || base.IsFullyInstantiated() {
			// Unnamed, universe, or instantiated types have no
			// object to reference.
			//
			// TODO(mcy): instantiated types could be encoded as a
			// base object plus type arguments.
			return ExportedSet{Unknown: true}
		}
		es.Members = append(es.Members, ExportedType{Depth: depth, Sym: sym})
	}
	return es
}

// An ImportedSet is one result slot's type set as decoded by the
// reader, with members already resolved to types.
type ImportedSet struct {
	Members []*types.Type
	HasNil  bool
	Unknown bool
}

// ImportResults records the result sets of an imported function.
func ImportResults(fn *ir.Func, sets []ImportedSet) {
	rts := make([]typeSet, len(sets))
	for i, is := range sets {
		if is.Unknown {
			rts[i] = typeSet{unknownType}
			continue
		}
		var set typeSet
		for _, t := range is.Members {
			set = set.Add(t)
		}
		if is.HasNil {
			set = set.Add(nilType)
		}
		rts[i] = set
	}
	fn.ResultTypeSets = rts
}

// ImportSplit records the devirtualized variant of an imported
// function, reconstructing its stub from fn's signature and result
// sets.
//
// It must run after [ImportResults] for fn.
func ImportSplit(fn *ir.Func, devirt []bool, paramNotes []string) {
	rts := fn.ResultTypeSets
	if rts == nil || len(devirt) != len(rts) {
		return
	}

	clone := func(f *types.Field, typ *types.Type) *types.Field {
		res := types.NewField(f.Pos, f.Sym, typ)
		res.SetIsDDD(f.IsDDD())
		return res
	}

	recvParams := fn.Type().RecvParams()
	params := make([]*types.Field, len(recvParams))
	for i, f := range recvParams {
		params[i] = clone(f, f.Type)
		if i < len(paramNotes) {
			params[i].Note = paramNotes[i]
		}
	}

	results := make([]*types.Field, len(fn.Type().Results()))
	for i, f := range fn.Type().Results() {
		typ := f.Type
		if devirt[i] {
			set := rts[i]
			if len(set) != 1 || set[0] == nilType || set[0] == unknownType {
				// The record and the sets disagree; drop the split
				// rather than guess.
				return
			}
			typ = set[0]
		}
		results[i] = clone(f, typ)
	}

	sym := fn.Sym().Pkg.Lookup(fn.Sym().Name + ".dv")
	fdv := ir.NewFunc(fn.Pos(), fn.Nname.Pos(), sym, types.NewSignature(nil, params, results))
	fdv.ABI = fn.ABI

	fn.DevirtVariant = fdv
	fdv.DevirtOriginal = fn
	fdv.ResultTypeSets = rts
	fdv.Inl = fn.Inl

	if concreteTypeDebug {
		base.Warn("ImportSplit(%v): variant %v", fn, fdv)
	}
}
