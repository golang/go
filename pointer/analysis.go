package pointer

// This file defines the entry points into the pointer analysis.

import (
	"fmt"
	"go/token"
	"io"
	"os"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/ssa"
)

// nodeid denotes a node.
// It is an index within analysis.nodes.
// We use small integers, not *node pointers, for many reasons:
// - they are smaller on 64-bit systems.
// - sets of them can be represented compactly in bitvectors or BDDs.
// - order matters; a field offset can be computed by simple addition.
type nodeid uint32

// node.flags bitmask values.
const (
	ntObject    = 1 << iota // start of an object (addressable memory location)
	ntInterface             // conctype node of interface object (=> ntObject)
	ntFunction              // identity node of function object (=> ntObject)
)

// A node is an equivalence class of memory locations.
// Nodes may be pointers, pointed-to locations, neither, or both.
type node struct {
	// flags is a bitset of the node type (nt*) flags defined above.
	flags uint32

	// Number of following words belonging to the same "object" allocation.
	// (Set by endObject.)  Zero for all other nodes.
	size uint32

	// The type of the field denoted by this node.  Non-aggregate,
	// unless this is an iface.conctype node (i.e. the thing
	// pointed to by an interface) in which case typ is that type.
	typ types.Type

	// data holds additional attributes of this node, depending on
	// its flags.
	//
	// If ntObject is set, data is the ssa.Value of the
	// instruction that allocated this memory, or nil if it was
	// implicit.
	//
	// Special cases:
	// - If ntInterface is also set, data will be a *ssa.MakeInterface.
	// - If ntFunction is also set, this node is the first word of a
	//   function block, and data is a *cgnode (not an ssa.Value)
	//   representing this function.
	data interface{}

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
	// - *typeAssertConstraint y=x.(T)
	// - *invokeConstraint     y=x.f(params...)
	complex constraintset
}

type constraint interface {
	String() string

	// Called by solver to prepare a constraint, e.g. to
	// - initialize a points-to set (addrConstraint).
	// - attach it to a pointer node (complex constraints).
	init(a *analysis)

	// solve is called for complex constraints when the pts for
	// the node to which they are attached has changed.
	solve(a *analysis, n *node, delta nodeset)
}

// dst = &src
// pts(dst) ⊇ {src}
// A base constraint used to initialize the solver's pt sets
type addrConstraint struct {
	dst nodeid
	src nodeid
}

// dst = src
// A simple constraint represented directly as a copyTo graph edge.
type copyConstraint struct {
	dst nodeid
	src nodeid
}

// dst = src[offset]
// A complex constraint attached to src (the pointer)
type loadConstraint struct {
	offset uint32
	dst    nodeid
	src    nodeid
}

// dst[offset] = src
// A complex constraint attached to dst (the pointer)
type storeConstraint struct {
	offset uint32
	dst    nodeid
	src    nodeid
}

// dst = &src.f  or  dst = &src[0]
// A complex constraint attached to dst (the pointer)
type offsetAddrConstraint struct {
	offset uint32
	dst    nodeid
	src    nodeid
}

// dst = src.(typ)
// A complex constraint attached to src (the interface).
type typeAssertConstraint struct {
	typ types.Type
	dst nodeid
	src nodeid
}

// src.method(params...)
// A complex constraint attached to iface.
type invokeConstraint struct {
	method *types.Func // the abstract method
	iface  nodeid      // the interface
	params nodeid      // the first parameter in the params/results block
}

// An analysis instance holds the state of a single pointer analysis problem.
type analysis struct {
	config          *Config                     // the client's control/observer interface
	prog            *ssa.Program                // the program being analyzed
	log             io.Writer                   // log stream; nil to disable
	panicNode       nodeid                      // sink for panic, source for recover
	nodes           []*node                     // indexed by nodeid
	flattenMemo     map[types.Type][]*fieldInfo // memoization of flatten()
	constraints     []constraint                // set of constraints
	callsites       []*callsite                 // all callsites
	genq            []*cgnode                   // queue of functions to generate constraints for
	intrinsics      map[*ssa.Function]intrinsic // non-nil values are summaries for intrinsic fns
	reflectValueObj types.Object                // type symbol for reflect.Value (if present)
	reflectRtypeObj types.Object                // type symbol for reflect.rtype (if present)
	reflectRtype    *types.Pointer              // *reflect.rtype
	funcObj         map[*ssa.Function]nodeid    // default function object for each func
	probes          map[*ssa.CallCommon]nodeid  // maps call to print() to argument variable
	valNode         map[ssa.Value]nodeid        // node for each ssa.Value
	work            worklist                    // solver's worklist
}

func (a *analysis) warnf(pos token.Pos, format string, args ...interface{}) {
	if Warn := a.config.Warn; Warn != nil {
		Warn(pos, format, args...)
	} else {
		fmt.Fprintf(os.Stderr, "%s: warning: ", a.prog.Fset.Position(pos))
		fmt.Fprintf(os.Stderr, format, args...)
		fmt.Fprintln(os.Stderr)
	}
}

// Analyze runs the pointer analysis with the scope and options
// specified by config, and returns the (synthetic) root of the callgraph.
//
func Analyze(config *Config) CallGraphNode {
	a := &analysis{
		config:      config,
		log:         config.Log,
		prog:        config.prog(),
		valNode:     make(map[ssa.Value]nodeid),
		flattenMemo: make(map[types.Type][]*fieldInfo),
		intrinsics:  make(map[*ssa.Function]intrinsic),
		funcObj:     make(map[*ssa.Function]nodeid),
		probes:      make(map[*ssa.CallCommon]nodeid),
		work:        makeMapWorklist(),
	}

	if reflect := a.prog.PackagesByPath["reflect"]; reflect != nil {
		a.reflectValueObj = reflect.Object.Scope().Lookup("Value")
		a.reflectRtypeObj = reflect.Object.Scope().Lookup("rtype")
		a.reflectRtype = types.NewPointer(a.reflectRtypeObj.Type())
	}

	if false {
		a.log = os.Stderr // for debugging crashes; extremely verbose
	}

	if a.log != nil {
		fmt.Fprintln(a.log, "======== NEW ANALYSIS ========")
	}

	root := a.generate()

	// ---------- Presolver ----------

	// TODO(adonovan): opt: presolver optimisations.

	// ---------- Solver ----------

	a.solve()

	if a.log != nil {
		// Dump solution.
		for i, n := range a.nodes {
			if n.pts != nil {
				fmt.Fprintf(a.log, "pts(n%d) = %s : %s\n", i, n.pts, n.typ)
			}
		}
	}

	// Notify the client of the callsites if they're interested.
	if CallSite := a.config.CallSite; CallSite != nil {
		for _, site := range a.callsites {
			CallSite(site)
		}
	}

	Call := a.config.Call
	for _, site := range a.callsites {
		for nid := range a.nodes[site.targets].pts {
			cgn := a.nodes[nid].data.(*cgnode)

			// Notify the client of the call graph, if
			// they're interested.
			if Call != nil {
				Call(site, site.caller, cgn)
			}

			// Warn about calls to non-intrinsic external functions.

			if fn := cgn.fn; fn.Blocks == nil && a.findIntrinsic(fn) == nil {
				a.warnf(site.Pos(), "unsound call to unknown intrinsic: %s", fn)
				a.warnf(fn.Pos(), " (declared here)")
			}
		}
	}

	return root
}
