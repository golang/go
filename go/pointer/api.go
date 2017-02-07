// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"bytes"
	"fmt"
	"go/token"
	"io"

	"golang.org/x/tools/container/intsets"
	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types/typeutil"
)

// A Config formulates a pointer analysis problem for Analyze().
type Config struct {
	// Mains contains the set of 'main' packages to analyze
	// Clients must provide the analysis with at least one
	// package defining a main() function.
	//
	// Non-main packages in the ssa.Program that are not
	// dependencies of any main package may still affect the
	// analysis result, because they contribute runtime types and
	// thus methods.
	// TODO(adonovan): investigate whether this is desirable.
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
	// TODO(adonovan): this API doesn't scale well for batch tools
	// that want to dump the entire solution.  Perhaps optionally
	// populate a map[*ssa.DebugRef]Pointer in the Result, one
	// entry per source expression.
	//
	Queries         map[ssa.Value]struct{}
	IndirectQueries map[ssa.Value]struct{}

	// If Log is non-nil, log messages are written to it.
	// Logging is extremely verbose.
	Log io.Writer
}

type track uint32

const (
	trackChan  track = 1 << iota // track 'chan' references
	trackMap                     // track 'map' references
	trackPtr                     // track regular pointers
	trackSlice                   // track slice references

	trackAll = ^track(0)
)

// AddQuery adds v to Config.Queries.
// Precondition: CanPoint(v.Type()).
// TODO(adonovan): consider returning a new Pointer for this query,
// which will be initialized during analysis.  That avoids the needs
// for the corresponding ssa.Value-keyed maps in Config and Result.
func (c *Config) AddQuery(v ssa.Value) {
	if !CanPoint(v.Type()) {
		panic(fmt.Sprintf("%s is not a pointer-like value: %s", v, v.Type()))
	}
	if c.Queries == nil {
		c.Queries = make(map[ssa.Value]struct{})
	}
	c.Queries[v] = struct{}{}
}

// AddQuery adds v to Config.IndirectQueries.
// Precondition: CanPoint(v.Type().Underlying().(*types.Pointer).Elem()).
func (c *Config) AddIndirectQuery(v ssa.Value) {
	if c.IndirectQueries == nil {
		c.IndirectQueries = make(map[ssa.Value]struct{})
	}
	if !CanPoint(mustDeref(v.Type())) {
		panic(fmt.Sprintf("%s is not the address of a pointer-like value: %s", v, v.Type()))
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
	CallGraph       *callgraph.Graph      // discovered call graph
	Queries         map[ssa.Value]Pointer // pts(v) for each v in Config.Queries.
	IndirectQueries map[ssa.Value]Pointer // pts(*v) for each v in Config.IndirectQueries.
	Warnings        []Warning             // warnings of unsoundness
}

// A Pointer is an equivalence class of pointer-like values.
//
// A Pointer doesn't have a unique type because pointers of distinct
// types may alias the same object.
//
type Pointer struct {
	a *analysis
	n nodeid
}

// A PointsToSet is a set of labels (locations or allocations).
type PointsToSet struct {
	a   *analysis // may be nil if pts is nil
	pts *nodeset
}

func (s PointsToSet) String() string {
	var buf bytes.Buffer
	buf.WriteByte('[')
	if s.pts != nil {
		var space [50]int
		for i, l := range s.pts.AppendTo(space[:0]) {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(s.a.labelFor(nodeid(l)).String())
		}
	}
	buf.WriteByte(']')
	return buf.String()
}

// PointsTo returns the set of labels that this points-to set
// contains.
func (s PointsToSet) Labels() []*Label {
	var labels []*Label
	if s.pts != nil {
		var space [50]int
		for _, l := range s.pts.AppendTo(space[:0]) {
			labels = append(labels, s.a.labelFor(nodeid(l)))
		}
	}
	return labels
}

// If this PointsToSet came from a Pointer of interface kind
// or a reflect.Value, DynamicTypes returns the set of dynamic
// types that it may contain.  (For an interface, they will
// always be concrete types.)
//
// The result is a mapping whose keys are the dynamic types to which
// it may point.  For each pointer-like key type, the corresponding
// map value is the PointsToSet for pointers of that type.
//
// The result is empty unless CanHaveDynamicTypes(T).
//
func (s PointsToSet) DynamicTypes() *typeutil.Map {
	var tmap typeutil.Map
	tmap.SetHasher(s.a.hasher)
	if s.pts != nil {
		var space [50]int
		for _, x := range s.pts.AppendTo(space[:0]) {
			ifaceObjId := nodeid(x)
			if !s.a.isTaggedObject(ifaceObjId) {
				continue // !CanHaveDynamicTypes(tDyn)
			}
			tDyn, v, indirect := s.a.taggedValue(ifaceObjId)
			if indirect {
				panic("indirect tagged object") // implement later
			}
			pts, ok := tmap.At(tDyn).(PointsToSet)
			if !ok {
				pts = PointsToSet{s.a, new(nodeset)}
				tmap.Set(tDyn, pts)
			}
			pts.pts.addAll(&s.a.nodes[v].solve.pts)
		}
	}
	return &tmap
}

// Intersects reports whether this points-to set and the
// argument points-to set contain common members.
func (x PointsToSet) Intersects(y PointsToSet) bool {
	if x.pts == nil || y.pts == nil {
		return false
	}
	// This takes Θ(|x|+|y|) time.
	var z intsets.Sparse
	z.Intersection(&x.pts.Sparse, &y.pts.Sparse)
	return !z.IsEmpty()
}

func (p Pointer) String() string {
	return fmt.Sprintf("n%d", p.n)
}

// PointsTo returns the points-to set of this pointer.
func (p Pointer) PointsTo() PointsToSet {
	if p.n == 0 {
		return PointsToSet{}
	}
	return PointsToSet{p.a, &p.a.nodes[p.n].solve.pts}
}

// MayAlias reports whether the receiver pointer may alias
// the argument pointer.
func (p Pointer) MayAlias(q Pointer) bool {
	return p.PointsTo().Intersects(q.PointsTo())
}

// DynamicTypes returns p.PointsTo().DynamicTypes().
func (p Pointer) DynamicTypes() *typeutil.Map {
	return p.PointsTo().DynamicTypes()
}
