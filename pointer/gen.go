// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file defines the constraint generation phase.

import (
	"fmt"
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/ssa"
)

var (
	tEface     = types.NewInterface(nil)
	tInvalid   = types.Typ[types.Invalid]
	tUnsafePtr = types.Typ[types.UnsafePointer]
)

// ---------- Node creation ----------

// nextNode returns the index of the next unused node.
func (a *analysis) nextNode() nodeid {
	return nodeid(len(a.nodes))
}

// addNodes creates nodes for all scalar elements in type typ, and
// returns the id of the first one, or zero if the type was
// analytically uninteresting.
//
// comment explains the origin of the nodes, as a debugging aid.
//
func (a *analysis) addNodes(typ types.Type, comment string) nodeid {
	id := a.nextNode()
	for _, fi := range a.flatten(typ) {
		a.addOneNode(fi.typ, comment, fi)
	}
	if id == a.nextNode() {
		return 0 // type contained no pointers
	}
	return id
}

// addOneNode creates a single node with type typ, and returns its id.
//
// typ should generally be scalar (except for tagged.T nodes
// and struct/array identity nodes).  Use addNodes for non-scalar types.
//
// comment explains the origin of the nodes, as a debugging aid.
// subelement indicates the subelement, e.g. ".a.b[*].c".
//
func (a *analysis) addOneNode(typ types.Type, comment string, subelement *fieldInfo) nodeid {
	id := a.nextNode()
	a.nodes = append(a.nodes, &node{typ: typ, subelement: subelement})
	if a.log != nil {
		fmt.Fprintf(a.log, "\tcreate n%d %s for %s%s\n",
			id, typ, comment, subelement.path())
	}
	return id
}

// setValueNode associates node id with the value v.
// TODO(adonovan): disambiguate v by its CallGraphNode, if it's a local.
func (a *analysis) setValueNode(v ssa.Value, id nodeid) {
	a.valNode[v] = id
	if a.log != nil {
		fmt.Fprintf(a.log, "\tval[%s] = n%d  (%T)\n", v.Name(), id, v)
	}

	// Record the (v, id) relation if the client has queried v.
	if indirect, ok := a.config.QueryValues[v]; ok {
		if indirect {
			tmp := a.addNodes(v.Type(), "query.indirect")
			a.load(tmp, id, a.sizeof(v.Type()))
			id = tmp
		}
		ptrs := a.config.QueryResults
		if ptrs == nil {
			ptrs = make(map[ssa.Value][]Pointer)
			a.config.QueryResults = ptrs
		}
		ptrs[v] = append(ptrs[v], ptr{a, id})
	}
}

// endObject marks the end of a sequence of calls to addNodes denoting
// a single object allocation.
//
// obj is the start node of the object, from a prior call to nextNode.
// Its size, flags and (optionally) data will be updated.
//
func (a *analysis) endObject(obj nodeid, cgn *cgnode, val ssa.Value) *object {
	// Ensure object is non-empty by padding;
	// the pad will be the object node.
	size := uint32(a.nextNode() - obj)
	if size == 0 {
		a.addOneNode(tInvalid, "padding", nil)
	}
	objNode := a.nodes[obj]
	o := &object{
		size: size, // excludes padding
		cgn:  cgn,
		val:  val,
	}
	objNode.obj = o
	if val != nil && a.log != nil {
		fmt.Fprintf(a.log, "\tobj[%s] = n%d\n", val, obj)
	}

	return o
}

// makeFunctionObject creates and returns a new function object for
// fn, and returns the id of its first node.  It also enqueues fn for
// subsequent constraint generation.
//
func (a *analysis) makeFunctionObject(fn *ssa.Function) nodeid {
	if a.log != nil {
		fmt.Fprintf(a.log, "\t---- makeFunctionObject %s\n", fn)
	}

	// obj is the function object (identity, params, results).
	obj := a.nextNode()
	cgn := &cgnode{fn: fn, obj: obj}
	sig := fn.Signature
	a.addOneNode(sig, "func.cgnode", nil) // (scalar with Signature type)
	if recv := sig.Recv(); recv != nil {
		a.addNodes(recv.Type(), "func.recv")
	}
	a.addNodes(sig.Params(), "func.params")
	a.addNodes(sig.Results(), "func.results")
	a.endObject(obj, cgn, fn).flags |= otFunction

	if a.log != nil {
		fmt.Fprintf(a.log, "\t----\n")
	}

	// Queue it up for constraint processing.
	a.genq = append(a.genq, cgn)

	return obj
}

// makeFunction creates the shared function object (aka contour) for
// function fn and returns a 'func' value node that points to it.
//
func (a *analysis) makeFunction(fn *ssa.Function) nodeid {
	obj := a.makeFunctionObject(fn)
	a.funcObj[fn] = obj

	var comment string
	if a.log != nil {
		comment = fn.String()
	}
	id := a.addOneNode(fn.Type(), comment, nil)
	a.addressOf(id, obj)
	return id
}

// makeGlobal creates the value node and object node for global g,
// and returns the value node.
//
// The value node represents the address of the global variable, and
// points to the object (and nothing else).
//
// The object consists of the global variable itself (conceptually,
// the BSS address).
//
func (a *analysis) makeGlobal(g *ssa.Global) nodeid {
	var comment string
	if a.log != nil {
		fmt.Fprintf(a.log, "\t---- makeGlobal %s\n", g)
		comment = g.FullName()
	}

	// The nodes representing the object itself.
	obj := a.nextNode()
	a.addNodes(mustDeref(g.Type()), "global")
	a.endObject(obj, nil, g)

	if a.log != nil {
		fmt.Fprintf(a.log, "\t----\n")
	}

	// The node representing the address of the global.
	id := a.addOneNode(g.Type(), comment, nil)
	a.addressOf(id, obj)

	return id
}

// makeConstant creates the value node and object node (if needed) for
// constant c, and returns the value node.
// An object node is created only for []byte or []rune constants.
// The value node points to the object node, iff present.
//
func (a *analysis) makeConstant(l *ssa.Const) nodeid {
	id := a.addNodes(l.Type(), "const")
	if !l.IsNil() {
		// []byte or []rune?
		if t, ok := l.Type().Underlying().(*types.Slice); ok {
			// Treat []T like *[1]T, 'make []T' like new([1]T).
			obj := a.nextNode()
			a.addNodes(sliceToArray(t), "array in slice constant")
			a.endObject(obj, nil, l)

			a.addressOf(id, obj)
		}
	}
	return id
}

// makeTagged creates a tagged object of type typ.
func (a *analysis) makeTagged(typ types.Type, cgn *cgnode, val ssa.Value) nodeid {
	obj := a.addOneNode(typ, "tagged.T", nil) // NB: type may be non-scalar!
	a.addNodes(typ, "tagged.v")
	a.endObject(obj, cgn, val).flags |= otTagged
	return obj
}

// makeRtype returns the canonical tagged object of type *rtype whose
// payload points to the sole rtype object for T.
func (a *analysis) makeRtype(T types.Type) nodeid {
	if v := a.rtypes.At(T); v != nil {
		return v.(nodeid)
	}

	// Create the object for the reflect.rtype itself, which is
	// ordinarily a large struct but here a single node will do.
	obj := a.nextNode()
	a.addOneNode(T, "reflect.rtype", nil)
	a.endObject(obj, nil, nil).rtype = T

	id := a.makeTagged(a.reflectRtypePtr, nil, nil)
	a.nodes[id].obj.rtype = T
	a.nodes[id+1].typ = T // trick (each *rtype tagged object is a singleton)
	a.addressOf(id+1, obj)

	a.rtypes.Set(T, id)
	return id
}

// rtypeValue returns the type of the *reflect.rtype-tagged object obj.
func (a *analysis) rtypeTaggedValue(obj nodeid) types.Type {
	tDyn, t, _ := a.taggedValue(obj)
	if tDyn != a.reflectRtypePtr {
		panic(fmt.Sprintf("not a *reflect.rtype-tagged value: obj=n%d tag=%v payload=n%d", obj, tDyn, t))
	}
	return a.nodes[t].typ
}

// valueNode returns the id of the value node for v, creating it (and
// the association) as needed.  It may return zero for uninteresting
// values containing no pointers.
//
// Nodes for locals are created en masse during genFunc and are
// implicitly contextualized by the function currently being analyzed
// (i.e. parameter to genFunc).
//
func (a *analysis) valueNode(v ssa.Value) nodeid {
	id, ok := a.valNode[v]
	if !ok {
		switch v := v.(type) {
		case *ssa.Function:
			id = a.makeFunction(v)

		case *ssa.Global:
			id = a.makeGlobal(v)

		case *ssa.Const:
			id = a.makeConstant(v)

		case *ssa.Capture:
			// TODO(adonovan): treat captures context-sensitively.
			id = a.addNodes(v.Type(), "capture")

		default:
			// *ssa.Parameters and ssa.Instruction values
			// are created by genFunc.
			// *Builtins are not true values.
			panic(v)
		}
		a.setValueNode(v, id)
	}
	return id
}

// valueOffsetNode ascertains the node for tuple/struct value v,
// then returns the node for its subfield #index.
//
func (a *analysis) valueOffsetNode(v ssa.Value, index int) nodeid {
	id := a.valueNode(v)
	if id == 0 {
		panic(fmt.Sprintf("cannot offset within n0: %s = %s", v.Name(), v))
	}
	return id + nodeid(a.offsetOf(v.Type(), index))
}

// taggedValue returns the dynamic type tag, the (first node of the)
// payload, and the indirect flag of the tagged object starting at id.
// It returns tDyn==nil if obj is not a tagged object.
//
func (a *analysis) taggedValue(id nodeid) (tDyn types.Type, v nodeid, indirect bool) {
	n := a.nodes[id]
	flags := n.obj.flags
	if flags&otTagged != 0 {
		return n.typ, id + 1, flags&otIndirect != 0
	}
	return
}

// funcParams returns the first node of the params block of the
// function whose object node (obj.flags&otFunction) is id.
//
func (a *analysis) funcParams(id nodeid) nodeid {
	n := a.nodes[id]
	if n.obj == nil || n.obj.flags&otFunction == 0 {
		panic(fmt.Sprintf("funcParams(n%d): not a function object block", id))
	}
	return id + 1
}

// funcResults returns the first node of the results block of the
// function whose object node (obj.flags&otFunction) is id.
//
func (a *analysis) funcResults(id nodeid) nodeid {
	n := a.nodes[id]
	if n.obj == nil || n.obj.flags&otFunction == 0 {
		panic(fmt.Sprintf("funcResults(n%d): not a function object block", id))
	}
	sig := n.typ.(*types.Signature)
	id += 1 + nodeid(a.sizeof(sig.Params()))
	if sig.Recv() != nil {
		id += nodeid(a.sizeof(sig.Recv().Type()))
	}
	return id
}

// ---------- Constraint creation ----------

// copy creates a constraint of the form dst = src.
// sizeof is the width (in logical fields) of the copied type.
//
func (a *analysis) copy(dst, src nodeid, sizeof uint32) {
	if src == dst || sizeof == 0 {
		return // trivial
	}
	if src == 0 || dst == 0 {
		panic(fmt.Sprintf("ill-typed copy dst=n%d src=n%d", dst, src))
	}
	for i := uint32(0); i < sizeof; i++ {
		a.addConstraint(&copyConstraint{dst, src})
		src++
		dst++
	}
}

// addressOf creates a constraint of the form id = &obj.
func (a *analysis) addressOf(id, obj nodeid) {
	if id == 0 {
		panic("addressOf: zero id")
	}
	if obj == 0 {
		panic("addressOf: zero obj")
	}
	a.addConstraint(&addrConstraint{id, obj})
}

// load creates a load constraint of the form dst = *src.
// sizeof is the width (in logical fields) of the loaded type.
//
func (a *analysis) load(dst, src nodeid, sizeof uint32) {
	a.loadOffset(dst, src, 0, sizeof)
}

// loadOffset creates a load constraint of the form dst = src[offset].
// offset is the pointer offset in logical fields.
// sizeof is the width (in logical fields) of the loaded type.
//
func (a *analysis) loadOffset(dst, src nodeid, offset uint32, sizeof uint32) {
	if dst == 0 {
		return // load of non-pointerlike value
	}
	if src == 0 && dst == 0 {
		return // non-pointerlike operation
	}
	if src == 0 || dst == 0 {
		panic(fmt.Sprintf("ill-typed load dst=n%d src=n%d", dst, src))
	}
	for i := uint32(0); i < sizeof; i++ {
		a.addConstraint(&loadConstraint{offset, dst, src})
		offset++
		dst++
	}
}

// store creates a store constraint of the form *dst = src.
// sizeof is the width (in logical fields) of the stored type.
//
func (a *analysis) store(dst, src nodeid, sizeof uint32) {
	a.storeOffset(dst, src, 0, sizeof)
}

// storeOffset creates a store constraint of the form dst[offset] = src.
// offset is the pointer offset in logical fields.
// sizeof is the width (in logical fields) of the stored type.
//
func (a *analysis) storeOffset(dst, src nodeid, offset uint32, sizeof uint32) {
	if src == 0 {
		return // store of non-pointerlike value
	}
	if src == 0 && dst == 0 {
		return // non-pointerlike operation
	}
	if src == 0 || dst == 0 {
		panic(fmt.Sprintf("ill-typed store dst=n%d src=n%d", dst, src))
	}
	for i := uint32(0); i < sizeof; i++ {
		a.addConstraint(&storeConstraint{offset, dst, src})
		offset++
		src++
	}
}

// offsetAddr creates an offsetAddr constraint of the form dst = &src.#offset.
// offset is the field offset in logical fields.
//
func (a *analysis) offsetAddr(dst, src nodeid, offset uint32) {
	if offset == 0 {
		// Simplify  dst = &src->f0
		//       to  dst = src
		// (NB: this optimisation is defeated by the identity
		// field prepended to struct and array objects.)
		a.copy(dst, src, 1)
	} else {
		a.addConstraint(&offsetAddrConstraint{offset, dst, src})
	}
}

// typeAssert creates a typeAssert constraint of the form dst = src.(T).
func (a *analysis) typeAssert(T types.Type, dst, src nodeid) {
	a.addConstraint(&typeAssertConstraint{T, dst, src})
}

// addConstraint adds c to the constraint set.
func (a *analysis) addConstraint(c constraint) {
	a.constraints = append(a.constraints, c)
	if a.log != nil {
		fmt.Fprintf(a.log, "\t%s\n", c)
	}
}

// copyElems generates load/store constraints for *dst = *src,
// where src and dst are slices or *arrays.
// (If pts(·) of either is a known singleton, this is suboptimal.)
//
func (a *analysis) copyElems(typ types.Type, dst, src nodeid) {
	tmp := a.addNodes(typ, "copy")
	sz := a.sizeof(typ)
	a.loadOffset(tmp, src, 1, sz)
	a.storeOffset(dst, tmp, 1, sz)
}

// ---------- Constraint generation ----------

// genConv generates constraints for the conversion operation conv.
func (a *analysis) genConv(conv *ssa.Convert, cgn *cgnode) {
	res := a.valueNode(conv)
	if res == 0 {
		return // result is non-pointerlike
	}

	tSrc := conv.X.Type()
	tDst := conv.Type()

	switch utSrc := tSrc.Underlying().(type) {
	case *types.Slice:
		// []byte/[]rune -> string?
		return

	case *types.Pointer:
		// *T -> unsafe.Pointer?
		if tDst == tUnsafePtr {
			// ignore for now
			// a.copy(res, a.valueNode(conv.X), 1)
			return
		}

	case *types.Basic:
		switch utDst := tDst.Underlying().(type) {
		case *types.Pointer:
			// unsafe.Pointer -> *T?  (currently unsound)
			if utSrc == tUnsafePtr {
				// For now, suppress unsafe.Pointer conversion
				// warnings on "syscall" package.
				// TODO(adonovan): audit for soundness.
				if conv.Parent().Pkg.Object.Path() != "syscall" {
					a.warnf(conv.Pos(),
						"unsound: %s contains an unsafe.Pointer conversion (to %s)",
						conv.Parent(), tDst)
				}

				// For now, we treat unsafe.Pointer->*T
				// conversion like new(T) and create an
				// unaliased object.  In future we may handle
				// unsafe conversions soundly; see TODO file.
				obj := a.addNodes(mustDeref(tDst), "unsafe.Pointer conversion")
				a.endObject(obj, cgn, conv)
				a.addressOf(res, obj)
				return
			}

		case *types.Slice:
			// string -> []byte/[]rune (or named aliases)?
			if utSrc.Info()&types.IsString != 0 {
				obj := a.addNodes(sliceToArray(tDst), "convert")
				a.endObject(obj, cgn, conv)
				a.addressOf(res, obj)
				return
			}

		case *types.Basic:
			// TODO(adonovan):
			// unsafe.Pointer -> uintptr?
			// uintptr -> unsafe.Pointer
			//
			// The language doesn't adequately specify the
			// behaviour of these operations, but almost
			// all uses of these conversions (even in the
			// spec) seem to imply a non-moving garbage
			// collection strategy, or implicit "pinning"
			// semantics for unsafe.Pointer conversions.

			// TODO(adonovan): we need more work before we can handle
			// cryptopointers well.
			if utSrc == tUnsafePtr || utDst == tUnsafePtr {
				// Ignore for now.  See TODO file for ideas.
				return
			}

			return // ignore all other basic type conversions
		}
	}

	panic(fmt.Sprintf("illegal *ssa.Convert %s -> %s: %s", tSrc, tDst, conv.Parent()))
}

// genAppend generates constraints for a call to append.
func (a *analysis) genAppend(instr *ssa.Call, cgn *cgnode) {
	// Consider z = append(x, y).   y is optional.
	// This may allocate a new [1]T array; call its object w.
	// We get the following constraints:
	// 	z = x
	// 	z = &w
	//     *z = *y

	x := a.valueNode(instr.Call.Args[0])

	z := a.valueNode(instr)
	a.copy(z, x, 1) // z = x

	if len(instr.Call.Args) == 1 {
		return // no allocation for z = append(x) or _ = append(x).
	}

	// TODO(adonovan): test append([]byte, ...string) []byte.

	y := a.valueNode(instr.Call.Args[1])
	tArray := sliceToArray(instr.Call.Args[0].Type())

	var w nodeid
	w = a.nextNode()
	a.addNodes(tArray, "append")
	a.endObject(w, cgn, instr)

	a.copyElems(tArray.Elem(), z, y) // *z = *y
	a.addressOf(z, w)                //  z = &w
}

// genBuiltinCall generates contraints for a call to a built-in.
func (a *analysis) genBuiltinCall(instr ssa.CallInstruction, cgn *cgnode) {
	call := instr.Common()
	switch call.Value.(*ssa.Builtin).Object().Name() {
	case "append":
		// Safe cast: append cannot appear in a go or defer statement.
		a.genAppend(instr.(*ssa.Call), cgn)

	case "copy":
		tElem := call.Args[0].Type().Underlying().(*types.Slice).Elem()
		a.copyElems(tElem, a.valueNode(call.Args[0]), a.valueNode(call.Args[1]))

	case "panic":
		a.copy(a.panicNode, a.valueNode(call.Args[0]), 1)

	case "recover":
		if v := instr.Value(); v != nil {
			a.copy(a.valueNode(v), a.panicNode, 1)
		}

	case "print":
		// Analytically print is a no-op, but it's a convenient hook
		// for testing the pts of an expression, so we notify the client.
		// Existing uses in Go core libraries are few and harmless.
		if Print := a.config.Print; Print != nil {
			// Due to context-sensitivity, we may encounter
			// the same print() call in many contexts, so
			// we merge them to a canonical node.
			probe := a.probes[call]
			t := call.Args[0].Type()

			// First time?  Create the canonical probe node.
			if probe == 0 {
				probe = a.addNodes(t, "print")
				a.probes[call] = probe
				Print(call, ptr{a, probe}) // notify client
			}

			a.copy(probe, a.valueNode(call.Args[0]), a.sizeof(t))
		}

	default:
		// No-ops: close len cap real imag complex println delete.
	}
}

// shouldUseContext defines the context-sensitivity policy.  It
// returns true if we should analyse all static calls to fn anew.
//
// Obviously this interface rather limits how much freedom we have to
// choose a policy.  The current policy, rather arbitrarily, is true
// for intrinsics and accessor methods (actually: short, single-block,
// call-free functions).  This is just a starting point.
//
func (a *analysis) shouldUseContext(fn *ssa.Function) bool {
	if a.findIntrinsic(fn) != nil {
		return true // treat intrinsics context-sensitively
	}
	if len(fn.Blocks) != 1 {
		return false // too expensive
	}
	blk := fn.Blocks[0]
	if len(blk.Instrs) > 10 {
		return false // too expensive
	}
	if fn.Synthetic != "" && (fn.Pkg == nil || fn != fn.Pkg.Func("init")) {
		return true // treat synthetic wrappers context-sensitively
	}
	for _, instr := range blk.Instrs {
		switch instr := instr.(type) {
		case ssa.CallInstruction:
			// Disallow function calls (except to built-ins)
			// because of the danger of unbounded recursion.
			if _, ok := instr.Common().Value.(*ssa.Builtin); !ok {
				return false
			}
		}
	}
	return true
}

// genStaticCall generates constraints for a statically dispatched
// function call.  It returns a node whose pts() will be the set of
// possible call targets (in this case, a singleton).
//
func (a *analysis) genStaticCall(call *ssa.CallCommon, result nodeid) nodeid {
	// Ascertain the context (contour/CGNode) for a particular call.
	var obj nodeid
	fn := call.StaticCallee()
	if a.shouldUseContext(fn) {
		obj = a.makeFunctionObject(fn) // new contour for this call
	} else {
		a.valueNode(fn)     // ensure shared contour was created
		obj = a.funcObj[fn] // ordinary (shared) contour.
	}

	sig := call.Signature()
	targets := a.addOneNode(sig, "call.targets", nil)
	a.addressOf(targets, obj) // (a singleton)

	// Copy receiver, if any.
	params := a.funcParams(obj)
	args := call.Args
	if sig.Recv() != nil {
		sz := a.sizeof(sig.Recv().Type())
		a.copy(params, a.valueNode(args[0]), sz)
		params += nodeid(sz)
		args = args[1:]
	}

	// Copy actual parameters into formal params block.
	// Must loop, since the actuals aren't contiguous.
	for i, arg := range args {
		sz := a.sizeof(sig.Params().At(i).Type())
		a.copy(params, a.valueNode(arg), sz)
		params += nodeid(sz)
	}

	// Copy formal results block to actual result.
	if result != 0 {
		a.copy(result, a.funcResults(obj), a.sizeof(sig.Results()))
	}

	return targets
}

// genDynamicCall generates constraints for a dynamic function call.
// It returns a node whose pts() will be the set of possible call targets.
//
func (a *analysis) genDynamicCall(call *ssa.CallCommon, result nodeid) nodeid {
	fn := a.valueNode(call.Value)
	sig := call.Signature()

	// We add dynamic closure rules that store the arguments into,
	// and load the results from, the P/R block of each function
	// discovered in pts(fn).

	var offset uint32 = 1 // P/R block starts at offset 1
	for i, arg := range call.Args {
		sz := a.sizeof(sig.Params().At(i).Type())
		a.storeOffset(fn, a.valueNode(arg), offset, sz)
		offset += sz
	}
	if result != 0 {
		a.loadOffset(result, fn, offset, a.sizeof(sig.Results()))
	}
	return fn
}

// genInvoke generates constraints for a dynamic method invocation.
// It returns a node whose pts() will be the set of possible call targets.
//
func (a *analysis) genInvoke(call *ssa.CallCommon, result nodeid) nodeid {
	if call.Value.Type() == a.reflectType {
		return a.genInvokeReflectType(call, result)
	}

	sig := call.Signature()

	// Allocate a contiguous targets/params/results block for this call.
	block := a.nextNode()
	targets := a.addOneNode(sig, "invoke.targets", nil)
	p := a.addNodes(sig.Params(), "invoke.params")
	r := a.addNodes(sig.Results(), "invoke.results")

	// Copy the actual parameters into the call's params block.
	for i, n := 0, sig.Params().Len(); i < n; i++ {
		sz := a.sizeof(sig.Params().At(i).Type())
		a.copy(p, a.valueNode(call.Args[i]), sz)
		p += nodeid(sz)
	}
	// Copy the call's results block to the actual results.
	if result != 0 {
		a.copy(result, r, a.sizeof(sig.Results()))
	}

	// We add a dynamic invoke constraint that will add
	// edges from the caller's P/R block to the callee's
	// P/R block for each discovered call target.
	a.addConstraint(&invokeConstraint{call.Method, a.valueNode(call.Value), block})

	return targets
}

// genInvokeReflectType is a specialization of genInvoke where the
// receiver type is a reflect.Type, under the assumption that there
// can be at most one implementation of this interface, *reflect.rtype.
//
// (Though this may appear to be an instance of a pattern---method
// calls on interfaces known to have exactly one implementation---in
// practice it occurs rarely, so we special case for reflect.Type.)
//
// In effect we treat this:
//    var rt reflect.Type = ...
//    rt.F()
// as this:
//    rt.(*reflect.rtype).F()
//
// It returns a node whose pts() will be the (singleton) set of
// possible call targets.
//
func (a *analysis) genInvokeReflectType(call *ssa.CallCommon, result nodeid) nodeid {
	// Unpack receiver into rtype
	rtype := a.addOneNode(a.reflectRtypePtr, "rtype.recv", nil)
	recv := a.valueNode(call.Value)
	a.typeAssert(a.reflectRtypePtr, rtype, recv)

	// Look up the concrete method.
	meth := a.reflectRtypePtr.MethodSet().Lookup(call.Method.Pkg(), call.Method.Name())
	fn := a.prog.Method(meth)

	obj := a.makeFunctionObject(fn) // new contour for this call

	// From now on, it's essentially a static call, but little is
	// gained by factoring together the code for both cases.

	sig := fn.Signature // concrete method
	targets := a.addOneNode(sig, "call.targets", nil)
	a.addressOf(targets, obj) // (a singleton)

	// Copy receiver.
	params := a.funcParams(obj)
	a.copy(params, rtype, 1)
	params++

	// Copy actual parameters into formal params block.
	// Must loop, since the actuals aren't contiguous.
	for i, arg := range call.Args {
		sz := a.sizeof(sig.Params().At(i).Type())
		a.copy(params, a.valueNode(arg), sz)
		params += nodeid(sz)
	}

	// Copy formal results block to actual result.
	if result != 0 {
		a.copy(result, a.funcResults(obj), a.sizeof(sig.Results()))
	}

	return obj
}

// genCall generates contraints for call instruction instr.
func (a *analysis) genCall(caller *cgnode, instr ssa.CallInstruction) {
	call := instr.Common()

	// Intrinsic implementations of built-in functions.
	if _, ok := call.Value.(*ssa.Builtin); ok {
		a.genBuiltinCall(instr, caller)
		return
	}

	var result nodeid
	if v := instr.Value(); v != nil {
		result = a.valueNode(v)
	}

	// The node whose pts(·) will contain all targets of the call.
	var targets nodeid
	switch {
	case call.StaticCallee() != nil:
		targets = a.genStaticCall(call, result)
	case call.IsInvoke():
		targets = a.genInvoke(call, result)
	default:
		targets = a.genDynamicCall(call, result)
	}

	site := &callsite{
		caller:  caller,
		targets: targets,
		instr:   instr,
		pos:     instr.Pos(),
	}
	a.callsites = append(a.callsites, site)
	if a.log != nil {
		fmt.Fprintf(a.log, "\t%s to targets %s from %s\n",
			site.Description(), site.targets, site.caller)
	}
}

// genInstr generates contraints for instruction instr in context cgn.
func (a *analysis) genInstr(cgn *cgnode, instr ssa.Instruction) {
	if a.log != nil {
		var prefix string
		if val, ok := instr.(ssa.Value); ok {
			prefix = val.Name() + " = "
		}
		fmt.Fprintf(a.log, "; %s%s\n", prefix, instr)
	}

	switch instr := instr.(type) {
	case *ssa.DebugRef:
		// no-op.

	case *ssa.UnOp:
		switch instr.Op {
		case token.ARROW: // <-x
			// We can ignore instr.CommaOk because the node we're
			// altering is always at zero offset relative to instr.
			a.load(a.valueNode(instr), a.valueNode(instr.X), a.sizeof(instr.Type()))

		case token.MUL: // *x
			a.load(a.valueNode(instr), a.valueNode(instr.X), a.sizeof(instr.Type()))

		default:
			// NOT, SUB, XOR: no-op.
		}

	case *ssa.BinOp:
		// All no-ops.

	case ssa.CallInstruction: // *ssa.Call, *ssa.Go, *ssa.Defer
		a.genCall(cgn, instr)

	case *ssa.ChangeType:
		a.copy(a.valueNode(instr), a.valueNode(instr.X), 1)

	case *ssa.Convert:
		a.genConv(instr, cgn)

	case *ssa.Extract:
		a.copy(a.valueNode(instr),
			a.valueOffsetNode(instr.Tuple, instr.Index),
			a.sizeof(instr.Type()))

	case *ssa.FieldAddr:
		a.offsetAddr(a.valueNode(instr), a.valueNode(instr.X),
			a.offsetOf(mustDeref(instr.X.Type()), instr.Field))

	case *ssa.IndexAddr:
		a.offsetAddr(a.valueNode(instr), a.valueNode(instr.X), 1)

	case *ssa.Field:
		a.copy(a.valueNode(instr),
			a.valueOffsetNode(instr.X, instr.Field),
			a.sizeof(instr.Type()))

	case *ssa.Index:
		a.copy(a.valueNode(instr), 1+a.valueNode(instr.X), a.sizeof(instr.Type()))

	case *ssa.Select:
		recv := a.valueOffsetNode(instr, 2) // instr : (index, recvOk, recv0, ... recv_n-1)
		for _, st := range instr.States {
			elemSize := a.sizeof(st.Chan.Type().Underlying().(*types.Chan).Elem())
			switch st.Dir {
			case ast.RECV:
				a.load(recv, a.valueNode(st.Chan), elemSize)
				recv++

			case ast.SEND:
				a.store(a.valueNode(st.Chan), a.valueNode(st.Send), elemSize)
			}
		}

	case *ssa.Ret:
		results := a.funcResults(cgn.obj)
		for _, r := range instr.Results {
			sz := a.sizeof(r.Type())
			a.copy(results, a.valueNode(r), sz)
			results += nodeid(sz)
		}

	case *ssa.Send:
		a.store(a.valueNode(instr.Chan), a.valueNode(instr.X), a.sizeof(instr.X.Type()))

	case *ssa.Store:
		a.store(a.valueNode(instr.Addr), a.valueNode(instr.Val), a.sizeof(instr.Val.Type()))

	case *ssa.Alloc:
		obj := a.nextNode()
		a.addNodes(mustDeref(instr.Type()), "alloc")
		a.endObject(obj, cgn, instr)
		a.addressOf(a.valueNode(instr), obj)

	case *ssa.MakeSlice:
		obj := a.nextNode()
		a.addNodes(sliceToArray(instr.Type()), "makeslice")
		a.endObject(obj, cgn, instr)
		a.addressOf(a.valueNode(instr), obj)

	case *ssa.MakeChan:
		obj := a.nextNode()
		a.addNodes(instr.Type().Underlying().(*types.Chan).Elem(), "makechan")
		a.endObject(obj, cgn, instr)
		a.addressOf(a.valueNode(instr), obj)

	case *ssa.MakeMap:
		obj := a.nextNode()
		tmap := instr.Type().Underlying().(*types.Map)
		a.addNodes(tmap.Key(), "makemap.key")
		a.addNodes(tmap.Elem(), "makemap.value")
		a.endObject(obj, cgn, instr)
		a.addressOf(a.valueNode(instr), obj)

	case *ssa.MakeInterface:
		tConc := instr.X.Type()
		// Create nodes and constraints for all methods of the type.
		// Ascertaining which will be needed is undecidable in general.
		mset := tConc.MethodSet()
		for i, n := 0, mset.Len(); i < n; i++ {
			a.valueNode(a.prog.Method(mset.At(i)))
		}

		obj := a.makeTagged(tConc, cgn, instr)

		// Copy the value into it, if nontrivial.
		if x := a.valueNode(instr.X); x != 0 {
			a.copy(obj+1, x, a.sizeof(tConc))
		}
		a.addressOf(a.valueNode(instr), obj)

	case *ssa.ChangeInterface:
		a.copy(a.valueNode(instr), a.valueNode(instr.X), 1)

	case *ssa.TypeAssert:
		a.typeAssert(instr.AssertedType, a.valueNode(instr), a.valueNode(instr.X))

	case *ssa.Slice:
		a.copy(a.valueNode(instr), a.valueNode(instr.X), 1)

	case *ssa.If, *ssa.Jump:
		// no-op.

	case *ssa.Phi:
		sz := a.sizeof(instr.Type())
		for _, e := range instr.Edges {
			a.copy(a.valueNode(instr), a.valueNode(e), sz)
		}

	case *ssa.MakeClosure:
		fn := instr.Fn.(*ssa.Function)
		a.copy(a.valueNode(instr), a.valueNode(fn), 1)
		// Free variables are treated like global variables.
		for i, b := range instr.Bindings {
			a.copy(a.valueNode(fn.FreeVars[i]), a.valueNode(b), a.sizeof(b.Type()))
		}

	case *ssa.RunDefers:
		// The analysis is flow insensitive, so we just "call"
		// defers as we encounter them.

	case *ssa.Range:
		// Do nothing.  Next{Iter: *ssa.Range} handles this case.

	case *ssa.Next:
		if !instr.IsString { // map
			// Assumes that Next is always directly applied to a Range result.
			theMap := instr.Iter.(*ssa.Range).X
			tMap := theMap.Type().Underlying().(*types.Map)
			ksize := a.sizeof(tMap.Key())
			vsize := a.sizeof(tMap.Elem())

			// Load from the map's (k,v) into the tuple's (ok, k, v).
			a.load(a.valueNode(instr)+1, a.valueNode(theMap), ksize+vsize)
		}

	case *ssa.Lookup:
		if tMap, ok := instr.X.Type().Underlying().(*types.Map); ok {
			// CommaOk can be ignored: field 0 is a no-op.
			ksize := a.sizeof(tMap.Key())
			vsize := a.sizeof(tMap.Elem())
			a.loadOffset(a.valueNode(instr), a.valueNode(instr.X), ksize, vsize)
		}

	case *ssa.MapUpdate:
		tmap := instr.Map.Type().Underlying().(*types.Map)
		ksize := a.sizeof(tmap.Key())
		vsize := a.sizeof(tmap.Elem())
		m := a.valueNode(instr.Map)
		a.store(m, a.valueNode(instr.Key), ksize)
		a.storeOffset(m, a.valueNode(instr.Value), ksize, vsize)

	case *ssa.Panic:
		a.copy(a.panicNode, a.valueNode(instr.X), 1)

	default:
		panic(fmt.Sprintf("unimplemented: %T", instr))
	}
}

// genRootCalls generates the synthetic root of the callgraph and the
// initial calls from it to the analysis scope, such as main, a test
// or a library.
//
func (a *analysis) genRootCalls() *cgnode {
	r := ssa.NewFunction("<root>", new(types.Signature), "root of callgraph")
	r.Prog = a.prog // hack.
	r.Enclosing = r // hack, so Function.String() doesn't crash
	r.String()      // (asserts that it doesn't crash)
	root := &cgnode{fn: r}

	// For each main package, call main.init(), main.main().
	for _, mainPkg := range a.config.Mains {
		main := mainPkg.Func("main")
		if main == nil {
			panic(fmt.Sprintf("%s has no main function", mainPkg))
		}

		targets := a.addOneNode(main.Signature, "root.targets", nil)
		site := &callsite{
			caller:  root,
			targets: targets,
		}
		a.callsites = append(a.callsites, site)
		for _, fn := range [2]*ssa.Function{mainPkg.Func("init"), main} {
			if a.log != nil {
				fmt.Fprintf(a.log, "\troot call to %s:\n", fn)
			}
			a.copy(targets, a.valueNode(fn), 1)
		}
	}

	return root
}

// genFunc generates constraints for function fn.
func (a *analysis) genFunc(cgn *cgnode) {
	fn := cgn.fn

	impl := a.findIntrinsic(fn)

	if a.log != nil {
		fmt.Fprintln(a.log)
		fmt.Fprintln(a.log)

		// Hack: don't display body if intrinsic.
		if impl != nil {
			fn2 := *cgn.fn // copy
			fn2.Locals = nil
			fn2.Blocks = nil
			fn2.DumpTo(a.log)
		} else {
			cgn.fn.DumpTo(a.log)
		}
	}

	if impl != nil {
		impl(a, cgn)
		return
	}

	if fn.Blocks == nil {
		// External function with no intrinsic treatment.
		// We'll warn about calls to such functions at the end.
		return
	}

	// The value nodes for the params are in the func object block.
	params := a.funcParams(cgn.obj)
	for _, p := range fn.Params {
		// TODO(adonovan): record the context (cgn) too.
		a.setValueNode(p, params)
		params += nodeid(a.sizeof(p.Type()))
	}

	// Free variables are treated like global variables:
	// the outer function sets them with MakeClosure;
	// the inner function accesses them with Capture.

	// Create value nodes for all value instructions.
	// (Clobbers any previous nodes from same fn in different context.)
	if a.log != nil {
		fmt.Fprintln(a.log, "; Creating instruction values")
	}
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			switch instr := instr.(type) {
			case *ssa.Range:
				// do nothing: it has a funky type.

			case ssa.Value:
				var comment string
				if a.log != nil {
					comment = instr.Name()
				}
				id := a.addNodes(instr.Type(), comment)
				// TODO(adonovan): record the context (cgn) too.
				a.setValueNode(instr, id)
			}
		}
	}

	// Generate constraints for instructions.
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			a.genInstr(cgn, instr)
		}
	}

	// (Instruction Values will hang around in the environment.)
}

// generate generates offline constraints for the entire program.
// It returns the synthetic root of the callgraph.
//
func (a *analysis) generate() *cgnode {
	// Create a dummy node since we use the nodeid 0 for
	// non-pointerlike variables.
	a.addNodes(tInvalid, "(zero)")

	// Create the global node for panic values.
	a.panicNode = a.addNodes(tEface, "panic")

	// Create nodes and constraints for all methods of reflect.rtype.
	// (Shared contours are used by dynamic calls to reflect.Type
	// methods---typically just String().)
	if rtype := a.reflectRtypePtr; rtype != nil {
		mset := rtype.MethodSet()
		for i, n := 0, mset.Len(); i < n; i++ {
			a.valueNode(a.prog.Method(mset.At(i)))
		}
	}

	root := a.genRootCalls()

	// Generate constraints for entire program.
	// (Actually just the RTA-reachable portion of the program.
	// See Bacon & Sweeney, OOPSLA'96).
	for len(a.genq) > 0 {
		cgn := a.genq[0]
		a.genq = a.genq[1:]
		a.genFunc(cgn)
	}

	return root
}
