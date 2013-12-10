// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file defines the main datatypes and Analyze function of the pointer analysis.

import (
	"fmt"
	"go/token"
	"io"
	"os"
	"reflect"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/go/types/typemap"
	"code.google.com/p/go.tools/ssa"
)

// object.flags bitmask values.
const (
	otTagged   = 1 << iota // type-tagged object
	otIndirect             // type-tagged object with indirect payload
	otFunction             // function object
)

// An object represents a contiguous block of memory to which some
// (generalized) pointer may point.
//
// (Note: most variables called 'obj' are not *objects but nodeids
// such that a.nodes[obj].obj != nil.)
//
type object struct {
	// flags is a bitset of the node type (ot*) flags defined above.
	flags uint32

	// Number of following nodes belonging to the same "object"
	// allocation.  Zero for all other nodes.
	size uint32

	// data describes this object; it has one of these types:
	//
	// ssa.Value	for an object allocated by an SSA operation.
	// types.Type	for an rtype instance object or *rtype-tagged object.
	// string	for an instrinsic object, e.g. the array behind os.Args.
	// nil		for an object allocated by an instrinsic.
	//		(cgn provides the identity of the intrinsic.)
	data interface{}

	// The call-graph node (=context) in which this object was allocated.
	// May be nil for global objects: Global, Const, some Functions.
	cgn *cgnode
}

// nodeid denotes a node.
// It is an index within analysis.nodes.
// We use small integers, not *node pointers, for many reasons:
// - they are smaller on 64-bit systems.
// - sets of them can be represented compactly in bitvectors or BDDs.
// - order matters; a field offset can be computed by simple addition.
type nodeid uint32

// A node is an equivalence class of memory locations.
// Nodes may be pointers, pointed-to locations, neither, or both.
//
// Nodes that are pointed-to locations ("labels") have an enclosing
// object (see analysis.enclosingObject).
//
type node struct {
	// If non-nil, this node is the start of an object
	// (addressable memory location).
	// The following obj.size words implicitly belong to the object;
	// they locate their object by scanning back.
	obj *object

	// The type of the field denoted by this node.  Non-aggregate,
	// unless this is an tagged.T node (i.e. the thing
	// pointed to by an interface) in which case typ is that type.
	typ types.Type

	// subelement indicates which directly embedded subelement of
	// an object of aggregate type (struct, tuple, array) this is.
	subelement *fieldInfo // e.g. ".a.b[*].c"

	// Points-to sets.
	pts     nodeset // points-to set of this node
	prevPts nodeset // pts(n) in previous iteration (for difference propagation)

	// Graph edges
	copyTo nodeset // simple copy constraint edges

	// Complex constraints attached to this node (x).
	// - *loadConstraint       y=*x
	// - *offsetAddrConstraint y=&x.f or y=&x[0]
	// - *storeConstraint      *x=z
	// - *typeFilterConstraint y=x.(I)
	// - *untagConstraint      y=x.(C)
	// - *invokeConstraint     y=x.f(params...)
	complex constraintset
}

type constraint interface {
	String() string

	// For a complex constraint, returns the nodeid of the pointer
	// to which it is attached.
	ptr() nodeid

	// solve is called for complex constraints when the pts for
	// the node to which they are attached has changed.
	solve(a *analysis, n *node, delta nodeset)
}

// dst = &src
// pts(dst) ⊇ {src}
// A base constraint used to initialize the solver's pt sets
type addrConstraint struct {
	dst nodeid // (ptr)
	src nodeid
}

// dst = src
// A simple constraint represented directly as a copyTo graph edge.
type copyConstraint struct {
	dst nodeid
	src nodeid // (ptr)
}

// dst = src[offset]
// A complex constraint attached to src (the pointer)
type loadConstraint struct {
	offset uint32
	dst    nodeid
	src    nodeid // (ptr)
}

// dst[offset] = src
// A complex constraint attached to dst (the pointer)
type storeConstraint struct {
	offset uint32
	dst    nodeid // (ptr)
	src    nodeid
}

// dst = &src.f  or  dst = &src[0]
// A complex constraint attached to dst (the pointer)
type offsetAddrConstraint struct {
	offset uint32
	dst    nodeid
	src    nodeid // (ptr)
}

// dst = src.(typ)  where typ is an interface
// A complex constraint attached to src (the interface).
// No representation change: pts(dst) and pts(src) contains tagged objects.
type typeFilterConstraint struct {
	typ types.Type // an interface type
	dst nodeid
	src nodeid // (ptr)
}

// dst = src.(typ)  where typ is a concrete type
// A complex constraint attached to src (the interface).
//
// If exact, only tagged objects identical to typ are untagged.
// If !exact, tagged objects assignable to typ are untagged too.
// The latter is needed for various reflect operators, e.g. Send.
//
// This entails a representation change:
// pts(src) contains tagged objects,
// pts(dst) contains their payloads.
type untagConstraint struct {
	typ   types.Type // a concrete type
	dst   nodeid
	src   nodeid // (ptr)
	exact bool
}

// src.method(params...)
// A complex constraint attached to iface.
type invokeConstraint struct {
	method *types.Func // the abstract method
	iface  nodeid      // (ptr) the interface
	params nodeid      // the first parameter in the params/results block
}

// An analysis instance holds the state of a single pointer analysis problem.
type analysis struct {
	config      *Config                     // the client's control/observer interface
	prog        *ssa.Program                // the program being analyzed
	log         io.Writer                   // log stream; nil to disable
	panicNode   nodeid                      // sink for panic, source for recover
	nodes       []*node                     // indexed by nodeid
	flattenMemo map[types.Type][]*fieldInfo // memoization of flatten()
	constraints []constraint                // set of constraints
	cgnodes     []*cgnode                   // all cgnodes
	genq        []*cgnode                   // queue of functions to generate constraints for
	intrinsics  map[*ssa.Function]intrinsic // non-nil values are summaries for intrinsic fns
	probes      map[*ssa.CallCommon]nodeid  // maps call to print() to argument variable
	globalval   map[ssa.Value]nodeid        // node for each global ssa.Value
	globalobj   map[ssa.Value]nodeid        // maps v to sole member of pts(v), if singleton
	localval    map[ssa.Value]nodeid        // node for each local ssa.Value
	localobj    map[ssa.Value]nodeid        // maps v to sole member of pts(v), if singleton
	work        worklist                    // solver's worklist
	result      *Result                     // results of the analysis

	// Reflection & intrinsics:
	hasher              typemap.Hasher // cache of type hashes
	reflectValueObj     types.Object   // type symbol for reflect.Value (if present)
	reflectValueCall    *ssa.Function  // (reflect.Value).Call
	reflectRtypeObj     types.Object   // *types.TypeName for reflect.rtype (if present)
	reflectRtypePtr     *types.Pointer // *reflect.rtype
	reflectType         *types.Named   // reflect.Type
	rtypes              typemap.M      // nodeid of canonical *rtype-tagged object for type T
	reflectZeros        typemap.M      // nodeid of canonical T-tagged object for zero value
	runtimeSetFinalizer *ssa.Function  // runtime.SetFinalizer
}

// enclosingObj returns the object (addressible memory object) that encloses node id.
// Panic ensues if that node does not belong to any object.
func (a *analysis) enclosingObj(id nodeid) *object {
	// Find previous node with obj != nil.
	for i := id; i >= 0; i-- {
		n := a.nodes[i]
		if obj := n.obj; obj != nil {
			if i+nodeid(obj.size) <= id {
				break // out of bounds
			}
			return obj
		}
	}
	panic("node has no enclosing object")
}

// labelFor returns the Label for node id.
// Panic ensues if that node is not addressable.
func (a *analysis) labelFor(id nodeid) *Label {
	return &Label{
		obj:        a.enclosingObj(id),
		subelement: a.nodes[id].subelement,
	}
}

func (a *analysis) warnf(pos token.Pos, format string, args ...interface{}) {
	a.result.Warnings = append(a.result.Warnings, Warning{pos, fmt.Sprintf(format, args...)})
}

// Analyze runs the pointer analysis with the scope and options
// specified by config, and returns the (synthetic) root of the callgraph.
//
func Analyze(config *Config) *Result {
	a := &analysis{
		config:      config,
		log:         config.Log,
		prog:        config.prog(),
		globalval:   make(map[ssa.Value]nodeid),
		globalobj:   make(map[ssa.Value]nodeid),
		flattenMemo: make(map[types.Type][]*fieldInfo),
		hasher:      typemap.MakeHasher(),
		intrinsics:  make(map[*ssa.Function]intrinsic),
		work:        makeMapWorklist(),
		result: &Result{
			Queries:         make(map[ssa.Value][]Pointer),
			IndirectQueries: make(map[ssa.Value][]Pointer),
			PrintCalls:      make(map[*ssa.CallCommon]Pointer),
		},
	}

	if false {
		a.log = os.Stderr // for debugging crashes; extremely verbose
	}

	if a.log != nil {
		fmt.Fprintln(a.log, "======== NEW ANALYSIS ========")
	}

	if reflect := a.prog.ImportedPackage("reflect"); reflect != nil {
		rV := reflect.Object.Scope().Lookup("Value")
		a.reflectValueObj = rV
		a.reflectValueCall = a.prog.Method(rV.Type().MethodSet().Lookup(nil, "Call"))
		a.reflectType = reflect.Object.Scope().Lookup("Type").Type().(*types.Named)
		a.reflectRtypeObj = reflect.Object.Scope().Lookup("rtype")
		a.reflectRtypePtr = types.NewPointer(a.reflectRtypeObj.Type())

		// Override flattening of reflect.Value, treating it like a basic type.
		tReflectValue := a.reflectValueObj.Type()
		a.flattenMemo[tReflectValue] = []*fieldInfo{{typ: tReflectValue}}

		a.rtypes.SetHasher(a.hasher)
		a.reflectZeros.SetHasher(a.hasher)
	}
	if runtime := a.prog.ImportedPackage("runtime"); runtime != nil {
		a.runtimeSetFinalizer = runtime.Func("SetFinalizer")
	}

	root := a.generate()

	if a.log != nil {
		// Show size of constraint system.
		counts := make(map[reflect.Type]int)
		for _, c := range a.constraints {
			counts[reflect.TypeOf(c)]++
		}
		fmt.Fprintf(a.log, "# constraints:\t%d\n", len(a.constraints))
		for t, n := range counts {
			fmt.Fprintf(a.log, "\t%s:\t%d\n", t, n)
		}
		fmt.Fprintf(a.log, "# nodes:\t%d\n", len(a.nodes))
	}

	//a.optimize()

	a.solve()

	if a.log != nil {
		// Dump solution.
		for i, n := range a.nodes {
			if n.pts != nil {
				fmt.Fprintf(a.log, "pts(n%d) = %s : %s\n", i, n.pts, n.typ)
			}
		}
	}

	// Add dynamic edges to call graph.
	for _, caller := range a.cgnodes {
		for _, site := range caller.sites {
			for callee := range a.nodes[site.targets].pts {
				a.callEdge(site, callee)
			}
		}
	}

	if a.config.BuildCallGraph {
		a.result.CallGraph = &cgraph{root, a.cgnodes}
	}

	return a.result
}

// callEdge is called for each edge in the callgraph.
// calleeid is the callee's object node (has otFunction flag).
//
func (a *analysis) callEdge(site *callsite, calleeid nodeid) {
	obj := a.nodes[calleeid].obj
	if obj.flags&otFunction == 0 {
		panic(fmt.Sprintf("callEdge %s -> n%d: not a function object", site, calleeid))
	}
	callee := obj.cgn

	if a.config.BuildCallGraph {
		site.callees = append(site.callees, callee)
	}

	if a.log != nil {
		fmt.Fprintf(a.log, "\tcall edge %s -> %s\n", site, callee)
	}

	// Warn about calls to non-intrinsic external functions.
	// TODO(adonovan): de-dup these messages.
	if fn := callee.fn; fn.Blocks == nil && a.findIntrinsic(fn) == nil {
		a.warnf(site.pos(), "unsound call to unknown intrinsic: %s", fn)
		a.warnf(fn.Pos(), " (declared here)")
	}
}
