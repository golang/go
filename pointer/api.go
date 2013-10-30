// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"fmt"
	"go/token"
	"io"

	"code.google.com/p/go.tools/call"
	"code.google.com/p/go.tools/go/types/typemap"
	"code.google.com/p/go.tools/ssa"
)

// A Config formulates a pointer analysis problem for Analyze().
type Config struct {
	// Mains contains the set of 'main' packages to analyze
	// Clients must provide the analysis with at least one
	// package defining a main() function.
	Mains []*ssa.Package

	// Reflection determines whether to handle reflection
	// operators soundly, which is currently rather slow since it
	// causes constraint to be generated during solving
	// proportional to the number of constraint variables, which
	// has not yet been reduced by presolver optimisation.
	Reflection bool

	// BuildCallGraph determines whether to construct a callgraph.
	// If enabled, the graph will be available in Result.CallGraph.
	BuildCallGraph bool

	// Print is invoked during the analysis for each discovered
	// call to the built-in print(x), providing a convenient way
	// to identify arbitrary expressions of interest in the tests.
	//
	// Pointer p may be saved until the analysis is complete, at
	// which point its methods provide access to the analysis
	// (The result of callings its methods within the Print
	// callback is undefined.)  p is nil if x is non-pointerlike.
	//
	Print func(site *ssa.CallCommon, p Pointer)

	// The client populates Queries[v] for each ssa.Value v of
	// interest.
	//
	// The boolean (Indirect) indicates whether to compute the
	// points-to set for v (false) or *v (true): the latter is
	// typically wanted for Values corresponding to source-level
	// lvalues, e.g. an *ssa.Global.
	//
	// The pointer analysis will populate the corresponding
	// Results.Queries value when it creates the pointer variable
	// for v or *v.  Upon completion the client can inspect that
	// map for the results.
	//
	// If a Value belongs to a function that the analysis treats
	// context-sensitively, the corresponding Results.Queries slice
	// may have multiple Pointers, one per distinct context.  Use
	// PointsToCombined to merge them.
	//
	Queries map[ssa.Value]Indirect

	// If Log is non-nil, log messages are written to it.
	// Logging is extremely verbose.
	Log io.Writer
}

type Indirect bool // map[ssa.Value]Indirect is not a set

func (c *Config) prog() *ssa.Program {
	for _, main := range c.Mains {
		return main.Prog
	}
	panic("empty scope")
}

type Warning struct {
	Pos     token.Pos
	Message string
}

// A Result contains the results of a pointer analysis.
//
// See Config for how to request the various Result components.
//
type Result struct {
	CallGraph call.Graph              // discovered call graph
	Queries   map[ssa.Value][]Pointer // points-to sets for queried ssa.Values
	Warnings  []Warning               // warnings of unsoundness
}

// A Pointer is an equivalence class of pointerlike values.
type Pointer interface {
	// PointsTo returns the points-to set of this pointer.
	PointsTo() PointsToSet

	// MayAlias reports whether the receiver pointer may alias
	// the argument pointer.
	MayAlias(Pointer) bool

	// Context returns the context of this pointer,
	// if it corresponds to a local variable.
	Context() call.GraphNode

	String() string
}

// A PointsToSet is a set of labels (locations or allocations).
//
type PointsToSet interface {
	// PointsTo returns the set of labels that this points-to set
	// contains.
	Labels() []*Label

	// Intersects reports whether this points-to set and the
	// argument points-to set contain common members.
	Intersects(PointsToSet) bool

	// If this PointsToSet came from a Pointer of interface kind
	// or a reflect.Value, DynamicTypes returns the set of dynamic
	// types that it may contain.  (For an interface, they will
	// always be concrete types.)
	//
	// The result is a mapping whose keys are the dynamic types to
	// which it may point.  For each pointer-like key type, the
	// corresponding map value is a set of pointer abstractions of
	// that dynamic type, represented as a []Pointer slice.  Use
	// PointsToCombined to merge them.
	//
	// The result is empty unless CanHaveDynamicTypes(T).
	//
	DynamicTypes() *typemap.M
}

// Union returns the set containing all the elements of each set in sets.
func Union(sets ...PointsToSet) PointsToSet {
	var union ptset
	for _, set := range sets {
		set := set.(ptset)
		union.a = set.a
		union.pts.addAll(set.pts)
	}
	return union
}

// PointsToCombined returns the combined points-to set of all the
// specified pointers.
func PointsToCombined(ptrs []Pointer) PointsToSet {
	var ptsets []PointsToSet
	for _, ptr := range ptrs {
		ptsets = append(ptsets, ptr.PointsTo())
	}
	return Union(ptsets...)
}

// ---- PointsToSet public interface

type ptset struct {
	a   *analysis // may be nil if pts is nil
	pts nodeset
}

func (s ptset) Labels() []*Label {
	var labels []*Label
	for l := range s.pts {
		labels = append(labels, s.a.labelFor(l))
	}
	return labels
}

func (s ptset) DynamicTypes() *typemap.M {
	var tmap typemap.M
	tmap.SetHasher(s.a.hasher)
	for ifaceObjId := range s.pts {
		if !s.a.isTaggedObject(ifaceObjId) {
			continue // !CanHaveDynamicTypes(tDyn)
		}
		tDyn, v, indirect := s.a.taggedValue(ifaceObjId)
		if indirect {
			panic("indirect tagged object") // implement later
		}
		prev, _ := tmap.At(tDyn).([]Pointer)
		tmap.Set(tDyn, append(prev, ptr{s.a, nil, v}))
	}
	return &tmap
}

func (x ptset) Intersects(y_ PointsToSet) bool {
	y := y_.(ptset)
	for l := range x.pts {
		if _, ok := y.pts[l]; ok {
			return true
		}
	}
	return false
}

// ---- Pointer public interface

// ptr adapts a node to the Pointer interface.
type ptr struct {
	a   *analysis
	cgn *cgnode
	n   nodeid // non-zero
}

func (p ptr) String() string {
	return fmt.Sprintf("n%d", p.n)
}

func (p ptr) Context() call.GraphNode {
	return p.cgn
}

func (p ptr) PointsTo() PointsToSet {
	return ptset{p.a, p.a.nodes[p.n].pts}
}

func (p ptr) MayAlias(q Pointer) bool {
	return p.PointsTo().Intersects(q.PointsTo())
}

func (p ptr) DynamicTypes() *typemap.M {
	return p.PointsTo().DynamicTypes()
}
