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
	"runtime/debug"

	"code.google.com/p/go.tools/go/callgraph"
	"code.google.com/p/go.tools/go/ssa"
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/go/types/typeutil"
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

// An analysis instance holds the state of a single pointer analysis problem.
type analysis struct {
	config      *Config                     // the client's control/observer interface
	prog        *ssa.Program                // the program being analyzed
	log         io.Writer                   // log stream; nil to disable
	panicNode   nodeid                      // sink for panic, source for recover
	nodes       []*node                     // indexed by nodeid
	flattenMemo map[types.Type][]*fieldInfo // memoization of flatten()
	trackTypes  map[types.Type]bool         // memoization of shouldTrack()
	constraints []constraint                // set of constraints
	cgnodes     []*cgnode                   // all cgnodes
	genq        []*cgnode                   // queue of functions to generate constraints for
	intrinsics  map[*ssa.Function]intrinsic // non-nil values are summaries for intrinsic fns
	globalval   map[ssa.Value]nodeid        // node for each global ssa.Value
	globalobj   map[ssa.Value]nodeid        // maps v to sole member of pts(v), if singleton
	localval    map[ssa.Value]nodeid        // node for each local ssa.Value
	localobj    map[ssa.Value]nodeid        // maps v to sole member of pts(v), if singleton
	work        worklist                    // solver's worklist
	result      *Result                     // results of the analysis
	track       track                       // pointerlike types whose aliasing we track

	// Reflection & intrinsics:
	hasher              typeutil.Hasher // cache of type hashes
	reflectValueObj     types.Object    // type symbol for reflect.Value (if present)
	reflectValueCall    *ssa.Function   // (reflect.Value).Call
	reflectRtypeObj     types.Object    // *types.TypeName for reflect.rtype (if present)
	reflectRtypePtr     *types.Pointer  // *reflect.rtype
	reflectType         *types.Named    // reflect.Type
	rtypes              typeutil.Map    // nodeid of canonical *rtype-tagged object for type T
	reflectZeros        typeutil.Map    // nodeid of canonical T-tagged object for zero value
	runtimeSetFinalizer *ssa.Function   // runtime.SetFinalizer
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
	msg := fmt.Sprintf(format, args...)
	if a.log != nil {
		fmt.Fprintf(a.log, "%s: warning: %s\n", a.prog.Fset.Position(pos), msg)
	}
	a.result.Warnings = append(a.result.Warnings, Warning{pos, msg})
}

// computeTrackBits sets a.track to the necessary 'track' bits for the pointer queries.
func (a *analysis) computeTrackBits() {
	var queryTypes []types.Type
	for v := range a.config.Queries {
		queryTypes = append(queryTypes, v.Type())
	}
	for v := range a.config.IndirectQueries {
		queryTypes = append(queryTypes, mustDeref(v.Type()))
	}
	for _, t := range queryTypes {
		switch t.Underlying().(type) {
		case *types.Chan:
			a.track |= trackChan
		case *types.Map:
			a.track |= trackMap
		case *types.Pointer:
			a.track |= trackPtr
		case *types.Slice:
			a.track |= trackSlice
		case *types.Interface:
			a.track = trackAll
			return
		}
		if rVObj := a.reflectValueObj; rVObj != nil && types.Identical(t, rVObj.Type()) {
			a.track = trackAll
			return
		}
	}
}

// Analyze runs the pointer analysis with the scope and options
// specified by config, and returns the (synthetic) root of the callgraph.
//
// Pointer analysis of a transitively closed well-typed program should
// always succeed.  An error can occur only due to an internal bug.
//
func Analyze(config *Config) (result *Result, err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("internal error in pointer analysis: %v (please report this bug)", p)
			fmt.Fprintln(os.Stderr, "Internal panic in pointer analysis:")
			debug.PrintStack()
		}
	}()

	a := &analysis{
		config:      config,
		log:         config.Log,
		prog:        config.prog(),
		globalval:   make(map[ssa.Value]nodeid),
		globalobj:   make(map[ssa.Value]nodeid),
		flattenMemo: make(map[types.Type][]*fieldInfo),
		trackTypes:  make(map[types.Type]bool),
		hasher:      typeutil.MakeHasher(),
		intrinsics:  make(map[*ssa.Function]intrinsic),
		work:        makeMapWorklist(),
		result: &Result{
			Queries:         make(map[ssa.Value]Pointer),
			IndirectQueries: make(map[ssa.Value]Pointer),
		},
	}

	if false {
		a.log = os.Stderr // for debugging crashes; extremely verbose
	}

	if a.log != nil {
		fmt.Fprintln(a.log, "======== NEW ANALYSIS ========")
	}

	// Pointer analysis requires a complete program for soundness.
	// Check to prevent accidental misconfiguration.
	for _, pkg := range a.prog.AllPackages() {
		// (This only checks that the package scope is complete,
		// not that func bodies exist, but it's a good signal.)
		if !pkg.Object.Complete() {
			return nil, fmt.Errorf(`pointer analysis requires a complete program yet package %q was incomplete (set loader.Config.SourceImports during loading)`, pkg.Object.Path())
		}
	}

	if reflect := a.prog.ImportedPackage("reflect"); reflect != nil {
		rV := reflect.Object.Scope().Lookup("Value")
		a.reflectValueObj = rV
		a.reflectValueCall = a.prog.LookupMethod(rV.Type(), nil, "Call")
		a.reflectType = reflect.Object.Scope().Lookup("Type").Type().(*types.Named)
		a.reflectRtypeObj = reflect.Object.Scope().Lookup("rtype")
		a.reflectRtypePtr = types.NewPointer(a.reflectRtypeObj.Type())

		// Override flattening of reflect.Value, treating it like a basic type.
		tReflectValue := a.reflectValueObj.Type()
		a.flattenMemo[tReflectValue] = []*fieldInfo{{typ: tReflectValue}}

		// Override shouldTrack of reflect.Value and *reflect.rtype.
		// Always track pointers of these types.
		a.trackTypes[tReflectValue] = true
		a.trackTypes[a.reflectRtypePtr] = true

		a.rtypes.SetHasher(a.hasher)
		a.reflectZeros.SetHasher(a.hasher)
	}
	if runtime := a.prog.ImportedPackage("runtime"); runtime != nil {
		a.runtimeSetFinalizer = runtime.Func("SetFinalizer")
	}
	a.computeTrackBits()

	a.generate()

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

	a.optimize()

	a.solve()

	// Create callgraph.Nodes in deterministic order.
	if cg := a.result.CallGraph; cg != nil {
		for _, caller := range a.cgnodes {
			cg.CreateNode(caller.fn)
		}
	}

	// Add dynamic edges to call graph.
	for _, caller := range a.cgnodes {
		for _, site := range caller.sites {
			for callee := range a.nodes[site.targets].pts {
				a.callEdge(caller, site, callee)
			}
		}
	}

	return a.result, nil
}

// callEdge is called for each edge in the callgraph.
// calleeid is the callee's object node (has otFunction flag).
//
func (a *analysis) callEdge(caller *cgnode, site *callsite, calleeid nodeid) {
	obj := a.nodes[calleeid].obj
	if obj.flags&otFunction == 0 {
		panic(fmt.Sprintf("callEdge %s -> n%d: not a function object", site, calleeid))
	}
	callee := obj.cgn

	if cg := a.result.CallGraph; cg != nil {
		// TODO(adonovan): opt: I would expect duplicate edges
		// (to wrappers) to arise due to the elimination of
		// context information, but I haven't observed any.
		// Understand this better.
		callgraph.AddEdge(cg.CreateNode(caller.fn), site.instr, cg.CreateNode(callee.fn))
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
