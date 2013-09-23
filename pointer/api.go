// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"fmt"
	"go/token"
	"io"

	"code.google.com/p/go.tools/go/types/typemap"
	"code.google.com/p/go.tools/ssa"
)

type Config struct {
	// -------- Scope of the analysis --------

	// Clients must provide the analysis with at least one package defining a main() function.
	Mains []*ssa.Package // set of 'main' packages to analyze
	root  *ssa.Function  // synthetic analysis root

	// Reflection determines whether to handle reflection
	// operators soundly, which is currently rather slow since it
	// causes constraint to be generated during solving
	// proportional to the number of constraint variables, which
	// has not yet been reduced by presolver optimisation.
	Reflection bool

	// -------- Optional callbacks invoked by the analysis --------

	// Call is invoked for each discovered call-graph edge.  The
	// call-graph is a multigraph over CallGraphNodes with edges
	// labelled by the CallSite that gives rise to the edge.
	// (The caller node is available as site.Caller())
	//
	// Clients that wish to construct a call graph may provide
	// CallGraph.AddEdge here.
	//
	// The callgraph may be context-sensitive, i.e. it may
	// distinguish separate calls to the same function depending
	// on the context.
	//
	Call func(site CallSite, callee CallGraphNode)

	// CallSite is invoked for each call-site encountered in the
	// program.
	//
	// The callgraph may be context-sensitive, i.e. it may
	// distinguish separate calls to the same function depending
	// on the context.
	//
	CallSite func(site CallSite)

	// Warn is invoked for each warning encountered by the analysis,
	// e.g. unknown external function, unsound use of unsafe.Pointer.
	// pos may be zero if the position is not known.
	Warn func(pos token.Pos, format string, args ...interface{})

	// Print is invoked during the analysis for each discovered
	// call to the built-in print(x).
	//
	// Pointer p may be saved until the analysis is complete, at
	// which point its methods provide access to the analysis
	// (The result of callings its methods within the Print
	// callback is undefined.)  p is nil if x is non-pointerlike.
	//
	// TODO(adonovan): this was a stop-gap measure for identifing
	// arbitrary expressions of interest in the tests.  Now that
	// ssa.ValueForExpr exists, we should use that instead.
	//
	Print func(site *ssa.CallCommon, p Pointer)

	// The client populates QueryValues[v] for each ssa.Value v
	// of interest.
	//
	// The boolean (Indirect) indicates whether to compute the
	// points-to set for v (false) or *v (true): the latter is
	// typically wanted for Values corresponding to source-level
	// lvalues, e.g. an *ssa.Global.
	//
	// The pointer analysis will populate the corresponding
	// QueryResults value when it creates the pointer variable
	// for v or *v.  Upon completion the client can inspect the
	// map for the results.
	//
	// If a Value belongs to a function that the analysis treats
	// context-sensitively, the corresponding QueryResults slice
	// may have multiple Pointers, one per distinct context.  Use
	// PointsToCombined to merge them.
	//
	// TODO(adonovan): refactor the API: separate all results of
	// Analyze() into a dedicated Result struct.
	//
	QueryValues  map[ssa.Value]Indirect
	QueryResults map[ssa.Value][]Pointer

	// -------- Other configuration options --------

	// If Log is non-nil, a log messages are written to it.
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

// A Pointer is an equivalence class of pointerlike values.
//
// TODO(adonovan): add a method
//    Context() CallGraphNode
// for pointers corresponding to local variables,
//
type Pointer interface {
	// PointsTo returns the points-to set of this pointer.
	PointsTo() PointsToSet

	// MayAlias reports whether the receiver pointer may alias
	// the argument pointer.
	MayAlias(Pointer) bool

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
		tDyn, v, indirect := s.a.taggedValue(ifaceObjId)
		if tDyn == nil {
			continue // !CanHaveDynamicTypes(tDyn)
		}
		if indirect {
			panic("indirect tagged object") // implement later
		}
		prev, _ := tmap.At(tDyn).([]Pointer)
		tmap.Set(tDyn, append(prev, ptr{s.a, v}))
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
	a *analysis
	n nodeid // non-zero
}

func (p ptr) String() string {
	return fmt.Sprintf("n%d", p.n)
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
