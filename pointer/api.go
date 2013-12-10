// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"bytes"
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

	// QueryPrintCalls causes the analysis to record (in
	// Result.PrintCalls) the points-to set of the first operand
	// of each discovered call to the built-in print(x), providing
	// a convenient way to identify arbitrary expressions of
	// interest in the tests.
	//
	QueryPrintCalls bool

	// The client populates Queries[v] or IndirectQueries[v]
	// for each ssa.Value v of interest, to request that the
	// points-to sets pts(v) or pts(*v) be computed.  If the
	// client needs both points-to sets, v may appear in both
	// maps.
	//
	// (IndirectQueries is typically used for Values corresponding
	// to source-level lvalues, e.g. an *ssa.Global.)
	//
	// The analysis populates the corresponding
	// Result.{Indirect,}Queries map when it creates the pointer
	// variable for v or *v.  Upon completion the client can
	// inspect that map for the results.
	//
	// If a Value belongs to a function that the analysis treats
	// context-sensitively, the corresponding Result.{Indirect,}Queries
	// slice may have multiple Pointers, one per distinct context.
	// Use PointsToCombined to merge them.
	//
	// TODO(adonovan): this API doesn't scale well for batch tools
	// that want to dump the entire solution.
	//
	// TODO(adonovan): need we distinguish contexts?  Current
	// clients always combine them.
	//
	Queries         map[ssa.Value]struct{}
	IndirectQueries map[ssa.Value]struct{}

	// If Log is non-nil, log messages are written to it.
	// Logging is extremely verbose.
	Log io.Writer
}

// AddQuery adds v to Config.Queries.
func (c *Config) AddQuery(v ssa.Value) {
	if c.Queries == nil {
		c.Queries = make(map[ssa.Value]struct{})
	}
	c.Queries[v] = struct{}{}
}

// AddQuery adds v to Config.IndirectQueries.
func (c *Config) AddIndirectQuery(v ssa.Value) {
	if c.IndirectQueries == nil {
		c.IndirectQueries = make(map[ssa.Value]struct{})
	}
	c.IndirectQueries[v] = struct{}{}
}

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
	CallGraph       call.Graph                  // discovered call graph
	Queries         map[ssa.Value][]Pointer     // pts(v) for each v in Config.Queries.
	IndirectQueries map[ssa.Value][]Pointer     // pts(*v) for each v in Config.IndirectQueries.
	Warnings        []Warning                   // warnings of unsoundness
	PrintCalls      map[*ssa.CallCommon]Pointer // pts(x) for each call to print(x)
}

// A Pointer is an equivalence class of pointerlike values.
//
// A pointer doesn't have a unique type because pointers of distinct
// types may alias the same object.
//
type Pointer struct {
	a   *analysis
	cgn *cgnode
	n   nodeid // non-zero
}

// A PointsToSet is a set of labels (locations or allocations).
type PointsToSet struct {
	a   *analysis // may be nil if pts is nil
	pts nodeset
}

// Union returns the set containing all the elements of each set in sets.
func Union(sets ...PointsToSet) PointsToSet {
	var union PointsToSet
	for _, set := range sets {
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

func (s PointsToSet) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "[")
	sep := ""
	for l := range s.pts {
		fmt.Fprintf(&buf, "%s%s", sep, s.a.labelFor(l))
		sep = ", "
	}
	fmt.Fprintf(&buf, "]")
	return buf.String()
}

// PointsTo returns the set of labels that this points-to set
// contains.
func (s PointsToSet) Labels() []*Label {
	var labels []*Label
	for l := range s.pts {
		labels = append(labels, s.a.labelFor(l))
	}
	return labels
}

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
func (s PointsToSet) DynamicTypes() *typemap.M {
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
		tmap.Set(tDyn, append(prev, Pointer{s.a, nil, v}))
	}
	return &tmap
}

// Intersects reports whether this points-to set and the
// argument points-to set contain common members.
func (x PointsToSet) Intersects(y PointsToSet) bool {
	for l := range x.pts {
		if _, ok := y.pts[l]; ok {
			return true
		}
	}
	return false
}

func (p Pointer) String() string {
	return fmt.Sprintf("n%d", p.n)
}

// Context returns the context of this pointer,
// if it corresponds to a local variable.
func (p Pointer) Context() call.GraphNode {
	return p.cgn
}

// PointsTo returns the points-to set of this pointer.
func (p Pointer) PointsTo() PointsToSet {
	return PointsToSet{p.a, p.a.nodes[p.n].pts}
}

// MayAlias reports whether the receiver pointer may alias
// the argument pointer.
func (p Pointer) MayAlias(q Pointer) bool {
	return p.PointsTo().Intersects(q.PointsTo())
}

// DynamicTypes returns p.PointsTo().DynamicTypes().
func (p Pointer) DynamicTypes() *typemap.M {
	return p.PointsTo().DynamicTypes()
}
