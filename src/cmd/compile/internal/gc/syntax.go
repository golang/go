// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// “Abstract” syntax representation.

package gc

// A Node is a single node in the syntax tree.
// Actually the syntax tree is a syntax DAG, because there is only one
// node with Op=ONAME for a given instance of a variable x.
// The same is true for Op=OTYPE and Op=OLITERAL.
type Node struct {
	// Tree structure.
	// Generic recursive walks should follow these fields.
	Left  *Node
	Right *Node
	Ninit Nodes
	Nbody Nodes
	List  Nodes
	Rlist Nodes

	// most nodes
	Type *Type
	Orig *Node // original form, for printing, and tracking copies of ONAMEs

	// func
	Func *Func

	// ONAME
	Name *Name

	Sym *Sym        // various
	E   interface{} // Opt or Val, see methods below

	// Various. Usually an offset into a struct. For example:
	// - ONAME nodes that refer to local variables use it to identify their stack frame position.
	// - ODOT, ODOTPTR, and OINDREGSP use it to indicate offset relative to their base address.
	// - OSTRUCTKEY uses it to store the named field's offset.
	// - OXCASE and OXFALL use it to validate the use of fallthrough.
	// - ONONAME uses it to store the current value of iota, see Node.Iota
	// Possibly still more uses. If you find any, document them.
	Xoffset int64

	Lineno int32

	Esc uint16 // EscXXX

	Op        Op
	Ullman    uint8 // sethi/ullman number
	Addable   bool  // addressable
	Etype     EType // op for OASOP, etype for OTYPE, exclam for export, 6g saved reg, ChanDir for OTCHAN, for OINDEXMAP 1=LHS,0=RHS
	Bounded   bool  // bounds check unnecessary
	NonNil    bool  // guaranteed to be non-nil
	Class     Class // PPARAM, PAUTO, PEXTERN, etc
	Embedded  uint8 // ODCLFIELD embedded type
	Colas     bool  // OAS resulting from :=
	Diag      bool  // already printed error about this
	Noescape  bool  // func arguments do not escape; TODO(rsc): move Noescape to Func struct (see CL 7360)
	Walkdef   uint8 // tracks state during typecheckdef; 2 == loop detected
	Typecheck uint8 // tracks state during typechecking; 2 == loop detected
	Local     bool
	IsStatic  bool // whether this Node will be converted to purely static data
	Initorder uint8
	Used      bool // for variable/label declared and not used error
	Isddd     bool // is the argument variadic
	Implicit  bool
	Addrtaken bool  // address taken, even if not moved to heap
	Assigned  bool  // is the variable ever assigned to
	Likely    int8  // likeliness of if statement
	hasVal    int8  // +1 for Val, -1 for Opt, 0 for not yet set
	flags     uint8 // TODO: store more bool fields in this flag field
}

// IsAutoTmp indicates if n was created by the compiler as a temporary,
// based on the setting of the .AutoTemp flag in n's Name.
func (n *Node) IsAutoTmp() bool {
	if n == nil || n.Op != ONAME {
		return false
	}
	return n.Name.AutoTemp
}

const (
	hasBreak = 1 << iota
	isClosureVar
	isOutputParamHeapAddr
	noInline // used internally by inliner to indicate that a function call should not be inlined; set for OCALLFUNC and OCALLMETH only
)

func (n *Node) HasBreak() bool {
	return n.flags&hasBreak != 0
}
func (n *Node) SetHasBreak(b bool) {
	if b {
		n.flags |= hasBreak
	} else {
		n.flags &^= hasBreak
	}
}
func (n *Node) isClosureVar() bool {
	return n.flags&isClosureVar != 0
}
func (n *Node) setIsClosureVar(b bool) {
	if b {
		n.flags |= isClosureVar
	} else {
		n.flags &^= isClosureVar
	}
}
func (n *Node) noInline() bool {
	return n.flags&noInline != 0
}
func (n *Node) setNoInline(b bool) {
	if b {
		n.flags |= noInline
	} else {
		n.flags &^= noInline
	}
}

func (n *Node) IsOutputParamHeapAddr() bool {
	return n.flags&isOutputParamHeapAddr != 0
}
func (n *Node) setIsOutputParamHeapAddr(b bool) {
	if b {
		n.flags |= isOutputParamHeapAddr
	} else {
		n.flags &^= isOutputParamHeapAddr
	}
}

// Val returns the Val for the node.
func (n *Node) Val() Val {
	if n.hasVal != +1 {
		return Val{}
	}
	return Val{n.E}
}

// SetVal sets the Val for the node, which must not have been used with SetOpt.
func (n *Node) SetVal(v Val) {
	if n.hasVal == -1 {
		Debug['h'] = 1
		Dump("have Opt", n)
		Fatalf("have Opt")
	}
	n.hasVal = +1
	n.E = v.U
}

// Opt returns the optimizer data for the node.
func (n *Node) Opt() interface{} {
	if n.hasVal != -1 {
		return nil
	}
	return n.E
}

// SetOpt sets the optimizer data for the node, which must not have been used with SetVal.
// SetOpt(nil) is ignored for Vals to simplify call sites that are clearing Opts.
func (n *Node) SetOpt(x interface{}) {
	if x == nil && n.hasVal >= 0 {
		return
	}
	if n.hasVal == +1 {
		Debug['h'] = 1
		Dump("have Val", n)
		Fatalf("have Val")
	}
	n.hasVal = -1
	n.E = x
}

func (n *Node) Iota() int64 {
	return n.Xoffset
}

func (n *Node) SetIota(x int64) {
	n.Xoffset = x
}

// Name holds Node fields used only by named nodes (ONAME, OPACK, OLABEL, some OLITERAL).
type Name struct {
	Pack      *Node  // real package for import . names
	Pkg       *Pkg   // pkg for OPACK nodes
	Heapaddr  *Node  // temp holding heap address of param (could move to Param?)
	Defn      *Node  // initializing assignment
	Curfn     *Node  // function for local variables
	Param     *Param // additional fields for ONAME
	Decldepth int32  // declaration loop depth, increased for every loop or label
	Vargen    int32  // unique name for ONAME within a function.  Function outputs are numbered starting at one.
	Funcdepth int32
	Readonly  bool
	Captured  bool // is the variable captured by a closure
	Byval     bool // is the variable captured by value or by reference
	Needzero  bool // if it contains pointers, needs to be zeroed on function entry
	Keepalive bool // mark value live across unknown assembly call
	AutoTemp  bool // is the variable a temporary (implies no dwarf info. reset if escapes to heap)
}

type Param struct {
	Ntype *Node

	// ONAME PAUTOHEAP
	Stackcopy *Node // the PPARAM/PPARAMOUT on-stack slot (moved func params only)

	// ONAME PPARAM
	Field *Field // TFIELD in arg struct

	// ONAME closure linkage
	// Consider:
	//
	//	func f() {
	//		x := 1 // x1
	//		func() {
	//			use(x) // x2
	//			func() {
	//				use(x) // x3
	//				--- parser is here ---
	//			}()
	//		}()
	//	}
	//
	// There is an original declaration of x and then a chain of mentions of x
	// leading into the current function. Each time x is mentioned in a new closure,
	// we create a variable representing x for use in that specific closure,
	// since the way you get to x is different in each closure.
	//
	// Let's number the specific variables as shown in the code:
	// x1 is the original x, x2 is when mentioned in the closure,
	// and x3 is when mentioned in the closure in the closure.
	//
	// We keep these linked (assume N > 1):
	//
	//   - x1.Defn = original declaration statement for x (like most variables)
	//   - x1.Innermost = current innermost closure x (in this case x3), or nil for none
	//   - x1.isClosureVar() = false
	//
	//   - xN.Defn = x1, N > 1
	//   - xN.isClosureVar() = true, N > 1
	//   - x2.Outer = nil
	//   - xN.Outer = x(N-1), N > 2
	//
	//
	// When we look up x in the symbol table, we always get x1.
	// Then we can use x1.Innermost (if not nil) to get the x
	// for the innermost known closure function,
	// but the first reference in a closure will find either no x1.Innermost
	// or an x1.Innermost with .Funcdepth < Funcdepth.
	// In that case, a new xN must be created, linked in with:
	//
	//     xN.Defn = x1
	//     xN.Outer = x1.Innermost
	//     x1.Innermost = xN
	//
	// When we finish the function, we'll process its closure variables
	// and find xN and pop it off the list using:
	//
	//     x1 := xN.Defn
	//     x1.Innermost = xN.Outer
	//
	// We leave xN.Innermost set so that we can still get to the original
	// variable quickly. Not shown here, but once we're
	// done parsing a function and no longer need xN.Outer for the
	// lexical x reference links as described above, closurebody
	// recomputes xN.Outer as the semantic x reference link tree,
	// even filling in x in intermediate closures that might not
	// have mentioned it along the way to inner closures that did.
	// See closurebody for details.
	//
	// During the eventual compilation, then, for closure variables we have:
	//
	//     xN.Defn = original variable
	//     xN.Outer = variable captured in next outward scope
	//                to make closure where xN appears
	//
	// Because of the sharding of pieces of the node, x.Defn means x.Name.Defn
	// and x.Innermost/Outer means x.Name.Param.Innermost/Outer.
	Innermost *Node
	Outer     *Node

	// OTYPE pragmas
	//
	// TODO: Should Func pragmas also be stored on the Name?
	Pragma Pragma
}

// Func holds Node fields used only with function-like nodes.
type Func struct {
	Shortname  *Node
	Enter      Nodes // for example, allocate and initialize memory for escaping parameters
	Exit       Nodes
	Cvars      Nodes   // closure params
	Dcl        []*Node // autodcl for this func/closure
	Inldcl     Nodes   // copy of dcl for use in inlining
	Closgen    int
	Outerfunc  *Node // outer function (for closure)
	FieldTrack map[*Sym]struct{}
	Ntype      *Node // signature
	Top        int   // top context (Ecall, Eproc, etc)
	Closure    *Node // OCLOSURE <-> ODCLFUNC
	Nname      *Node

	Inl     Nodes // copy of the body for use in inlining
	InlCost int32
	Depth   int32

	Label int32 // largest auto-generated label in this function

	Endlineno int32
	WBLineno  int32 // line number of first write barrier

	Pragma          Pragma // go:xxx function annotations
	Dupok           bool   // duplicate definitions ok
	Wrapper         bool   // is method wrapper
	Needctxt        bool   // function uses context register (has closure variables)
	ReflectMethod   bool   // function calls reflect.Type.Method or MethodByName
	IsHiddenClosure bool
	NoFramePointer  bool // Must not use a frame pointer for this function
}

type Op uint8

// Node ops.
const (
	OXXX = Op(iota)

	// names
	ONAME    // var, const or func name
	ONONAME  // unnamed arg or return value: f(int, string) (int, error) { etc }
	OTYPE    // type name
	OPACK    // import
	OLITERAL // literal

	// expressions
	OADD             // Left + Right
	OSUB             // Left - Right
	OOR              // Left | Right
	OXOR             // Left ^ Right
	OADDSTR          // +{List} (string addition, list elements are strings)
	OADDR            // &Left
	OANDAND          // Left && Right
	OAPPEND          // append(List)
	OARRAYBYTESTR    // Type(Left) (Type is string, Left is a []byte)
	OARRAYBYTESTRTMP // Type(Left) (Type is string, Left is a []byte, ephemeral)
	OARRAYRUNESTR    // Type(Left) (Type is string, Left is a []rune)
	OSTRARRAYBYTE    // Type(Left) (Type is []byte, Left is a string)
	OSTRARRAYBYTETMP // Type(Left) (Type is []byte, Left is a string, ephemeral)
	OSTRARRAYRUNE    // Type(Left) (Type is []rune, Left is a string)
	OAS              // Left = Right or (if Colas=true) Left := Right
	OAS2             // List = Rlist (x, y, z = a, b, c)
	OAS2FUNC         // List = Rlist (x, y = f())
	OAS2RECV         // List = Rlist (x, ok = <-c)
	OAS2MAPR         // List = Rlist (x, ok = m["foo"])
	OAS2DOTTYPE      // List = Rlist (x, ok = I.(int))
	OASOP            // Left Etype= Right (x += y)
	OASWB            // Left = Right (with write barrier)
	OCALL            // Left(List) (function call, method call or type conversion)
	OCALLFUNC        // Left(List) (function call f(args))
	OCALLMETH        // Left(List) (direct method call x.Method(args))
	OCALLINTER       // Left(List) (interface method call x.Method(args))
	OCALLPART        // Left.Right (method expression x.Method, not called)
	OCAP             // cap(Left)
	OCLOSE           // close(Left)
	OCLOSURE         // func Type { Body } (func literal)
	OCMPIFACE        // Left Etype Right (interface comparison, x == y or x != y)
	OCMPSTR          // Left Etype Right (string comparison, x == y, x < y, etc)
	OCOMPLIT         // Right{List} (composite literal, not yet lowered to specific form)
	OMAPLIT          // Type{List} (composite literal, Type is map)
	OSTRUCTLIT       // Type{List} (composite literal, Type is struct)
	OARRAYLIT        // Type{List} (composite literal, Type is array)
	OSLICELIT        // Type{List} (composite literal, Type is slice)
	OPTRLIT          // &Left (left is composite literal)
	OCONV            // Type(Left) (type conversion)
	OCONVIFACE       // Type(Left) (type conversion, to interface)
	OCONVNOP         // Type(Left) (type conversion, no effect)
	OCOPY            // copy(Left, Right)
	ODCL             // var Left (declares Left of type Left.Type)

	// Used during parsing but don't last.
	ODCLFUNC  // func f() or func (r) f()
	ODCLFIELD // struct field, interface field, or func/method argument/return value.
	ODCLCONST // const pi = 3.14
	ODCLTYPE  // type Int int

	ODELETE    // delete(Left, Right)
	ODOT       // Left.Sym (Left is of struct type)
	ODOTPTR    // Left.Sym (Left is of pointer to struct type)
	ODOTMETH   // Left.Sym (Left is non-interface, Right is method name)
	ODOTINTER  // Left.Sym (Left is interface, Right is method name)
	OXDOT      // Left.Sym (before rewrite to one of the preceding)
	ODOTTYPE   // Left.Right or Left.Type (.Right during parsing, .Type once resolved)
	ODOTTYPE2  // Left.Right or Left.Type (.Right during parsing, .Type once resolved; on rhs of OAS2DOTTYPE)
	OEQ        // Left == Right
	ONE        // Left != Right
	OLT        // Left < Right
	OLE        // Left <= Right
	OGE        // Left >= Right
	OGT        // Left > Right
	OIND       // *Left
	OINDEX     // Left[Right] (index of array or slice)
	OINDEXMAP  // Left[Right] (index of map)
	OKEY       // Left:Right (key:value in struct/array/map literal)
	OSTRUCTKEY // Sym:Left (key:value in struct literal, after type checking)
	OLEN       // len(Left)
	OMAKE      // make(List) (before type checking converts to one of the following)
	OMAKECHAN  // make(Type, Left) (type is chan)
	OMAKEMAP   // make(Type, Left) (type is map)
	OMAKESLICE // make(Type, Left, Right) (type is slice)
	OMUL       // Left * Right
	ODIV       // Left / Right
	OMOD       // Left % Right
	OLSH       // Left << Right
	ORSH       // Left >> Right
	OAND       // Left & Right
	OANDNOT    // Left &^ Right
	ONEW       // new(Left)
	ONOT       // !Left
	OCOM       // ^Left
	OPLUS      // +Left
	OMINUS     // -Left
	OOROR      // Left || Right
	OPANIC     // panic(Left)
	OPRINT     // print(List)
	OPRINTN    // println(List)
	OPAREN     // (Left)
	OSEND      // Left <- Right
	OSLICE     // Left[List[0] : List[1]] (Left is untypechecked or slice)
	OSLICEARR  // Left[List[0] : List[1]] (Left is array)
	OSLICESTR  // Left[List[0] : List[1]] (Left is string)
	OSLICE3    // Left[List[0] : List[1] : List[2]] (Left is untypedchecked or slice)
	OSLICE3ARR // Left[List[0] : List[1] : List[2]] (Left is array)
	ORECOVER   // recover()
	ORECV      // <-Left
	ORUNESTR   // Type(Left) (Type is string, Left is rune)
	OSELRECV   // Left = <-Right.Left: (appears as .Left of OCASE; Right.Op == ORECV)
	OSELRECV2  // List = <-Right.Left: (apperas as .Left of OCASE; count(List) == 2, Right.Op == ORECV)
	OIOTA      // iota
	OREAL      // real(Left)
	OIMAG      // imag(Left)
	OCOMPLEX   // complex(Left, Right)
	OALIGNOF   // unsafe.Alignof(Left)
	OOFFSETOF  // unsafe.Offsetof(Left)
	OSIZEOF    // unsafe.Sizeof(Left)

	// statements
	OBLOCK    // { List } (block of code)
	OBREAK    // break
	OCASE     // case Left or List[0]..List[1]: Nbody (select case after processing; Left==nil and List==nil means default)
	OXCASE    // case List: Nbody (select case before processing; List==nil means default)
	OCONTINUE // continue
	ODEFER    // defer Left (Left must be call)
	OEMPTY    // no-op (empty statement)
	OFALL     // fallthrough (after processing)
	OXFALL    // fallthrough (before processing)
	OFOR      // for Ninit; Left; Right { Nbody }
	OGOTO     // goto Left
	OIF       // if Ninit; Left { Nbody } else { Rlist }
	OLABEL    // Left:
	OPROC     // go Left (Left must be call)
	ORANGE    // for List = range Right { Nbody }
	ORETURN   // return List
	OSELECT   // select { List } (List is list of OXCASE or OCASE)
	OSWITCH   // switch Ninit; Left { List } (List is a list of OXCASE or OCASE)
	OTYPESW   // List = Left.(type) (appears as .Left of OSWITCH)

	// types
	OTCHAN   // chan int
	OTMAP    // map[string]int
	OTSTRUCT // struct{}
	OTINTER  // interface{}
	OTFUNC   // func()
	OTARRAY  // []int, [8]int, [N]int or [...]int

	// misc
	ODDD        // func f(args ...int) or f(l...) or var a = [...]int{0, 1, 2}.
	ODDDARG     // func f(args ...int), introduced by escape analysis.
	OINLCALL    // intermediary representation of an inlined call.
	OEFACE      // itable and data words of an empty-interface value.
	OITAB       // itable word of an interface value.
	OIDATA      // data word of an interface value in Left
	OSPTR       // base pointer of a slice or string.
	OCLOSUREVAR // variable reference at beginning of closure function
	OCFUNC      // reference to c function pointer (not go func value)
	OCHECKNIL   // emit code to ensure pointer/interface not nil
	OVARKILL    // variable is dead
	OVARLIVE    // variable is alive
	OINDREGSP   // offset plus indirect of REGSP, such as 8(SP).

	// arch-specific opcodes
	OCMP    // compare: ACMP.
	ODEC    // decrement: ADEC.
	OINC    // increment: AINC.
	OEXTEND // extend: ACWD/ACDQ/ACQO.
	OHMUL   // high mul: AMUL/AIMUL for unsigned/signed (OMUL uses AIMUL for both).
	OLROT   // left rotate: AROL.
	ORROTC  // right rotate-carry: ARCR.
	ORETJMP // return to other function
	OPS     // compare parity set (for x86 NaN check)
	OPC     // compare parity clear (for x86 NaN check)
	OSQRT   // sqrt(float64), on systems that have hw support
	OGETG   // runtime.getg() (read g pointer)

	OEND
)

// Nodes is a pointer to a slice of *Node.
// For fields that are not used in most nodes, this is used instead of
// a slice to save space.
type Nodes struct{ slice *[]*Node }

// Slice returns the entries in Nodes as a slice.
// Changes to the slice entries (as in s[i] = n) will be reflected in
// the Nodes.
func (n Nodes) Slice() []*Node {
	if n.slice == nil {
		return nil
	}
	return *n.slice
}

// Len returns the number of entries in Nodes.
func (n Nodes) Len() int {
	if n.slice == nil {
		return 0
	}
	return len(*n.slice)
}

// Index returns the i'th element of Nodes.
// It panics if n does not have at least i+1 elements.
func (n Nodes) Index(i int) *Node {
	return (*n.slice)[i]
}

// First returns the first element of Nodes (same as n.Index(0)).
// It panics if n has no elements.
func (n Nodes) First() *Node {
	return (*n.slice)[0]
}

// Second returns the second element of Nodes (same as n.Index(1)).
// It panics if n has fewer than two elements.
func (n Nodes) Second() *Node {
	return (*n.slice)[1]
}

// Set sets n to a slice.
// This takes ownership of the slice.
func (n *Nodes) Set(s []*Node) {
	if len(s) == 0 {
		n.slice = nil
	} else {
		// Copy s and take address of t rather than s to avoid
		// allocation in the case where len(s) == 0 (which is
		// over 3x more common, dynamically, for make.bash).
		t := s
		n.slice = &t
	}
}

// Set1 sets n to a slice containing a single node.
func (n *Nodes) Set1(node *Node) {
	n.slice = &[]*Node{node}
}

// Set2 sets n to a slice containing two nodes.
func (n *Nodes) Set2(n1, n2 *Node) {
	n.slice = &[]*Node{n1, n2}
}

// MoveNodes sets n to the contents of n2, then clears n2.
func (n *Nodes) MoveNodes(n2 *Nodes) {
	n.slice = n2.slice
	n2.slice = nil
}

// SetIndex sets the i'th element of Nodes to node.
// It panics if n does not have at least i+1 elements.
func (n Nodes) SetIndex(i int, node *Node) {
	(*n.slice)[i] = node
}

// Addr returns the address of the i'th element of Nodes.
// It panics if n does not have at least i+1 elements.
func (n Nodes) Addr(i int) **Node {
	return &(*n.slice)[i]
}

// Append appends entries to Nodes.
// If a slice is passed in, this will take ownership of it.
func (n *Nodes) Append(a ...*Node) {
	if len(a) == 0 {
		return
	}
	if n.slice == nil {
		n.slice = &a
	} else {
		*n.slice = append(*n.slice, a...)
	}
}

// Prepend prepends entries to Nodes.
// If a slice is passed in, this will take ownership of it.
func (n *Nodes) Prepend(a ...*Node) {
	if len(a) == 0 {
		return
	}
	if n.slice == nil {
		n.slice = &a
	} else {
		*n.slice = append(a, *n.slice...)
	}
}

// AppendNodes appends the contents of *n2 to n, then clears n2.
func (n *Nodes) AppendNodes(n2 *Nodes) {
	switch {
	case n2.slice == nil:
	case n.slice == nil:
		n.slice = n2.slice
	default:
		*n.slice = append(*n.slice, *n2.slice...)
	}
	n2.slice = nil
}
