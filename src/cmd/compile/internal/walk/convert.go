// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"go/constant"
	"internal/binary"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

// walkConv walks an OCONV or OCONVNOP (but not OCONVIFACE) node.
func walkConv(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(n.X, init)
	if n.Op() == ir.OCONVNOP && n.Type() == n.X.Type() {
		return n.X
	}
	if n.Op() == ir.OCONVNOP && ir.ShouldCheckPtr(ir.CurFunc, 1) {
		if n.Type().IsUnsafePtr() && n.X.Type().IsUintptr() { // uintptr to unsafe.Pointer
			return walkCheckPtrArithmetic(n, init)
		}
	}
	param, result := rtconvfn(n.X.Type(), n.Type())
	if param == types.Txxx {
		return n
	}
	fn := types.BasicTypeNames[param] + "to" + types.BasicTypeNames[result]
	return typecheck.Conv(mkcall(fn, types.Types[result], init, typecheck.Conv(n.X, types.Types[param])), n.Type())
}

// walkConvInterface walks an OCONVIFACE node.
func walkConvInterface(n *ir.ConvExpr, init *ir.Nodes) ir.Node {

	n.X = walkExpr(n.X, init)

	fromType := n.X.Type()
	toType := n.Type()
	if !fromType.IsInterface() && !ir.IsBlank(ir.CurFunc.Nname) {
		// skip unnamed functions (func _())
		if fromType.HasShape() {
			// Unified IR uses OCONVIFACE for converting all derived types
			// to interface type. Avoid assertion failure in
			// MarkTypeUsedInInterface, because we've marked used types
			// separately anyway.
		} else {
			reflectdata.MarkTypeUsedInInterface(fromType, ir.CurFunc.LSym)
		}
	}

	if !fromType.IsInterface() {
		typeWord := reflectdata.ConvIfaceTypeWord(base.Pos, n)
		l := ir.NewBinaryExpr(base.Pos, ir.OMAKEFACE, typeWord, dataWord(n, init))
		l.SetType(toType)
		l.SetTypecheck(n.Typecheck())
		return l
	}
	if fromType.IsEmptyInterface() {
		base.Fatalf("OCONVIFACE can't operate on an empty interface")
	}

	// Evaluate the input interface.
	c := typecheck.TempAt(base.Pos, ir.CurFunc, fromType)
	init.Append(ir.NewAssignStmt(base.Pos, c, n.X))

	if toType.IsEmptyInterface() {
		// Implement interface to empty interface conversion:
		//
		// var res *uint8
		// res = (*uint8)(unsafe.Pointer(itab))
		// if res != nil {
		//    res = res.type
		// }

		// Grab its parts.
		itab := ir.NewUnaryExpr(base.Pos, ir.OITAB, c)
		itab.SetType(types.Types[types.TUINTPTR].PtrTo())
		itab.SetTypecheck(1)
		data := ir.NewUnaryExpr(n.Pos(), ir.OIDATA, c)
		data.SetType(types.Types[types.TUINT8].PtrTo()) // Type is generic pointer - we're just passing it through.
		data.SetTypecheck(1)

		typeWord := typecheck.TempAt(base.Pos, ir.CurFunc, types.NewPtr(types.Types[types.TUINT8]))
		init.Append(ir.NewAssignStmt(base.Pos, typeWord, typecheck.Conv(typecheck.Conv(itab, types.Types[types.TUNSAFEPTR]), typeWord.Type())))
		nif := ir.NewIfStmt(base.Pos, typecheck.Expr(ir.NewBinaryExpr(base.Pos, ir.ONE, typeWord, typecheck.NodNil())), nil, nil)
		nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, typeWord, itabType(typeWord))}
		init.Append(nif)

		// Build the result.
		// e = iface{typeWord, data}
		e := ir.NewBinaryExpr(base.Pos, ir.OMAKEFACE, typeWord, data)
		e.SetType(toType) // assign type manually, typecheck doesn't understand OEFACE.
		e.SetTypecheck(1)
		return e
	}

	// Must be converting I2I (more specific to less specific interface).
	// Use the same code as e, _ = c.(T).
	var rhs ir.Node
	if n.TypeWord == nil || n.TypeWord.Op() == ir.OADDR && n.TypeWord.(*ir.AddrExpr).X.Op() == ir.OLINKSYMOFFSET {
		// Fixed (not loaded from a dictionary) type.
		ta := ir.NewTypeAssertExpr(base.Pos, c, toType)
		ta.SetOp(ir.ODOTTYPE2)
		// Allocate a descriptor for this conversion to pass to the runtime.
		ta.Descriptor = makeTypeAssertDescriptor(toType, true)
		rhs = ta
	} else {
		ta := ir.NewDynamicTypeAssertExpr(base.Pos, ir.ODYNAMICDOTTYPE2, c, n.TypeWord)
		rhs = ta
	}
	rhs.SetType(toType)
	rhs.SetTypecheck(1)

	res := typecheck.TempAt(base.Pos, ir.CurFunc, toType)
	as := ir.NewAssignListStmt(base.Pos, ir.OAS2DOTTYPE, []ir.Node{res, ir.BlankNode}, []ir.Node{rhs})
	init.Append(as)
	return res
}

// Returns the data word (the second word) used to represent conv.X in
// an interface.
func dataWord(conv *ir.ConvExpr, init *ir.Nodes) ir.Node {
	pos, n := conv.Pos(), conv.X
	fromType := n.Type()

	// If it's a pointer, it is its own representation.
	if types.IsDirectIface(fromType) {
		return n
	}

	isInteger := fromType.IsInteger()
	isBool := fromType.IsBoolean()
	if sc := fromType.SoleComponent(); sc != nil {
		isInteger = sc.IsInteger()
		isBool = sc.IsBoolean()
	}
	// Try a bunch of cases to avoid an allocation.
	var value ir.Node
	switch {
	case fromType.Size() == 0:
		// n is zero-sized. Use zerobase.
		cheapExpr(n, init) // Evaluate n for side-effects. See issue 19246.
		value = ir.NewLinksymExpr(base.Pos, ir.Syms.Zerobase, types.Types[types.TUINTPTR])
	case isBool || fromType.Size() == 1 && isInteger:
		// n is a bool/byte. Use staticuint64s[n * 8] on little-endian
		// and staticuint64s[n * 8 + 7] on big-endian.
		n = cheapExpr(n, init)
		n = soleComponent(init, n)
		// byteindex widens n so that the multiplication doesn't overflow.
		index := ir.NewBinaryExpr(base.Pos, ir.OLSH, byteindex(n), ir.NewInt(base.Pos, 3))
		if ssagen.Arch.LinkArch.ByteOrder == binary.BigEndian {
			index = ir.NewBinaryExpr(base.Pos, ir.OADD, index, ir.NewInt(base.Pos, 7))
		}
		// The actual type is [256]uint64, but we use [256*8]uint8 so we can address
		// individual bytes.
		staticuint64s := ir.NewLinksymExpr(base.Pos, ir.Syms.Staticuint64s, types.NewArray(types.Types[types.TUINT8], 256*8))
		xe := ir.NewIndexExpr(base.Pos, staticuint64s, index)
		xe.SetBounded(true)
		value = xe
	case n.Op() == ir.ONAME && n.(*ir.Name).Class == ir.PEXTERN && n.(*ir.Name).Readonly():
		// n is a readonly global; use it directly.
		value = n
	case conv.Esc() == ir.EscNone && fromType.Size() <= 1024:
		// n does not escape. Use a stack temporary initialized to n.
		value = typecheck.TempAt(base.Pos, ir.CurFunc, fromType)
		init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, value, n)))
	}
	if value != nil {
		// The interface data word is &value.
		return typecheck.Expr(typecheck.NodAddr(value))
	}

	// Time to do an allocation. We'll call into the runtime for that.
	fnname, argType, needsaddr := dataWordFuncName(fromType)
	var fn *ir.Name

	var args []ir.Node
	if needsaddr {
		// Types of large or unknown size are passed by reference.
		// Orderexpr arranged for n to be a temporary for all
		// the conversions it could see. Comparison of an interface
		// with a non-interface, especially in a switch on interface value
		// with non-interface cases, is not visible to order.stmt, so we
		// have to fall back on allocating a temp here.
		if !ir.IsAddressable(n) {
			n = copyExpr(n, fromType, init)
		}
		fn = typecheck.LookupRuntime(fnname, fromType)
		args = []ir.Node{reflectdata.ConvIfaceSrcRType(base.Pos, conv), typecheck.NodAddr(n)}
	} else {
		// Use a specialized conversion routine that takes the type being
		// converted by value, not by pointer.
		fn = typecheck.LookupRuntime(fnname)
		var arg ir.Node
		switch {
		case fromType == argType:
			// already in the right type, nothing to do
			arg = n
		case fromType.Kind() == argType.Kind(),
			fromType.IsPtrShaped() && argType.IsPtrShaped():
			// can directly convert (e.g. named type to underlying type, or one pointer to another)
			// TODO: never happens because pointers are directIface?
			arg = ir.NewConvExpr(pos, ir.OCONVNOP, argType, n)
		case fromType.IsInteger() && argType.IsInteger():
			// can directly convert (e.g. int32 to uint32)
			arg = ir.NewConvExpr(pos, ir.OCONV, argType, n)
		default:
			// unsafe cast through memory
			arg = copyExpr(n, fromType, init)
			var addr ir.Node = typecheck.NodAddr(arg)
			addr = ir.NewConvExpr(pos, ir.OCONVNOP, argType.PtrTo(), addr)
			arg = ir.NewStarExpr(pos, addr)
			arg.SetType(argType)
		}
		args = []ir.Node{arg}
	}
	call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, nil)
	call.Args = args
	return safeExpr(walkExpr(typecheck.Expr(call), init), init)
}

// walkBytesRunesToString walks an OBYTES2STR or ORUNES2STR node.
func walkBytesRunesToString(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	a := typecheck.NodNil()
	if n.Esc() == ir.EscNone {
		// Create temporary buffer for string on stack.
		a = stackBufAddr(tmpstringbufsize, types.Types[types.TUINT8])
	}
	if n.Op() == ir.ORUNES2STR {
		// slicerunetostring(*[32]byte, []rune) string
		return mkcall("slicerunetostring", n.Type(), init, a, n.X)
	}
	// slicebytetostring(*[32]byte, ptr *byte, n int) string
	n.X = cheapExpr(n.X, init)
	ptr, len := backingArrayPtrLen(n.X)
	return mkcall("slicebytetostring", n.Type(), init, a, ptr, len)
}

// walkBytesToStringTemp walks an OBYTES2STRTMP node.
func walkBytesToStringTemp(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	n.X = walkExpr(n.X, init)
	if !base.Flag.Cfg.Instrumenting {
		// Let the backend handle OBYTES2STRTMP directly
		// to avoid a function call to slicebytetostringtmp.
		return n
	}
	// slicebytetostringtmp(ptr *byte, n int) string
	n.X = cheapExpr(n.X, init)
	ptr, len := backingArrayPtrLen(n.X)
	return mkcall("slicebytetostringtmp", n.Type(), init, ptr, len)
}

// walkRuneToString walks an ORUNESTR node.
func walkRuneToString(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	a := typecheck.NodNil()
	if n.Esc() == ir.EscNone {
		a = stackBufAddr(4, types.Types[types.TUINT8])
	}
	// intstring(*[4]byte, rune)
	return mkcall("intstring", n.Type(), init, a, typecheck.Conv(n.X, types.Types[types.TINT64]))
}

// walkStringToBytes walks an OSTR2BYTES node.
func walkStringToBytes(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	s := n.X
	if ir.IsConst(s, constant.String) {
		sc := ir.StringVal(s)

		// Allocate a [n]byte of the right size.
		t := types.NewArray(types.Types[types.TUINT8], int64(len(sc)))
		var a ir.Node
		if n.Esc() == ir.EscNone && len(sc) <= int(ir.MaxImplicitStackVarSize) {
			a = stackBufAddr(t.NumElem(), t.Elem())
		} else {
			types.CalcSize(t)
			a = ir.NewUnaryExpr(base.Pos, ir.ONEW, nil)
			a.SetType(types.NewPtr(t))
			a.SetTypecheck(1)
			a.MarkNonNil()
		}
		p := typecheck.TempAt(base.Pos, ir.CurFunc, t.PtrTo()) // *[n]byte
		init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, p, a)))

		// Copy from the static string data to the [n]byte.
		if len(sc) > 0 {
			sptr := ir.NewUnaryExpr(base.Pos, ir.OSPTR, s)
			sptr.SetBounded(true)
			as := ir.NewAssignStmt(base.Pos, ir.NewStarExpr(base.Pos, p), ir.NewStarExpr(base.Pos, typecheck.ConvNop(sptr, t.PtrTo())))
			appendWalkStmt(init, as)
		}

		// Slice the [n]byte to a []byte.
		slice := ir.NewSliceExpr(n.Pos(), ir.OSLICEARR, p, nil, nil, nil)
		slice.SetType(n.Type())
		slice.SetTypecheck(1)
		return walkExpr(slice, init)
	}

	a := typecheck.NodNil()
	if n.Esc() == ir.EscNone {
		// Create temporary buffer for slice on stack.
		a = stackBufAddr(tmpstringbufsize, types.Types[types.TUINT8])
	}
	// stringtoslicebyte(*32[byte], string) []byte
	return mkcall("stringtoslicebyte", n.Type(), init, a, typecheck.Conv(s, types.Types[types.TSTRING]))
}

// walkStringToBytesTemp walks an OSTR2BYTESTMP node.
func walkStringToBytesTemp(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	// []byte(string) conversion that creates a slice
	// referring to the actual string bytes.
	// This conversion is handled later by the backend and
	// is only for use by internal compiler optimizations
	// that know that the slice won't be mutated.
	// The only such case today is:
	// for i, c := range []byte(string)
	n.X = walkExpr(n.X, init)
	return n
}

// walkStringToRunes walks an OSTR2RUNES node.
func walkStringToRunes(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	a := typecheck.NodNil()
	if n.Esc() == ir.EscNone {
		// Create temporary buffer for slice on stack.
		a = stackBufAddr(tmpstringbufsize, types.Types[types.TINT32])
	}
	// stringtoslicerune(*[32]rune, string) []rune
	return mkcall("stringtoslicerune", n.Type(), init, a, typecheck.Conv(n.X, types.Types[types.TSTRING]))
}

// dataWordFuncName returns the name of the function used to convert a value of type "from"
// to the data word of an interface.
// argType is the type the argument needs to be coerced to.
// needsaddr reports whether the value should be passed (needaddr==false) or its address (needsaddr==true).
func dataWordFuncName(from *types.Type) (fnname string, argType *types.Type, needsaddr bool) {
	if from.IsInterface() {
		base.Fatalf("can only handle non-interfaces")
	}
	switch {
	case from.Size() == 2 && uint8(from.Alignment()) == 2:
		return "convT16", types.Types[types.TUINT16], false
	case from.Size() == 4 && uint8(from.Alignment()) == 4 && !from.HasPointers():
		return "convT32", types.Types[types.TUINT32], false
	case from.Size() == 8 && uint8(from.Alignment()) == uint8(types.Types[types.TUINT64].Alignment()) && !from.HasPointers():
		return "convT64", types.Types[types.TUINT64], false
	}
	if sc := from.SoleComponent(); sc != nil {
		switch {
		case sc.IsString():
			return "convTstring", types.Types[types.TSTRING], false
		case sc.IsSlice():
			return "convTslice", types.NewSlice(types.Types[types.TUINT8]), false // the element type doesn't matter
		}
	}

	if from.HasPointers() {
		return "convT", types.Types[types.TUNSAFEPTR], true
	}
	return "convTnoptr", types.Types[types.TUNSAFEPTR], true
}

// rtconvfn returns the parameter and result types that will be used by a
// runtime function to convert from type src to type dst. The runtime function
// name can be derived from the names of the returned types.
//
// If no such function is necessary, it returns (Txxx, Txxx).
func rtconvfn(src, dst *types.Type) (param, result types.Kind) {
	if ssagen.Arch.SoftFloat {
		return types.Txxx, types.Txxx
	}

	switch ssagen.Arch.LinkArch.Family {
	case sys.ARM, sys.MIPS:
		if src.IsFloat() {
			switch dst.Kind() {
			case types.TINT64, types.TUINT64:
				return types.TFLOAT64, dst.Kind()
			}
		}
		if dst.IsFloat() {
			switch src.Kind() {
			case types.TINT64, types.TUINT64:
				return src.Kind(), dst.Kind()
			}
		}

	case sys.I386:
		if src.IsFloat() {
			switch dst.Kind() {
			case types.TINT64, types.TUINT64:
				return types.TFLOAT64, dst.Kind()
			case types.TUINT32, types.TUINT, types.TUINTPTR:
				return types.TFLOAT64, types.TUINT32
			}
		}
		if dst.IsFloat() {
			switch src.Kind() {
			case types.TINT64, types.TUINT64:
				return src.Kind(), dst.Kind()
			case types.TUINT32, types.TUINT, types.TUINTPTR:
				return types.TUINT32, types.TFLOAT64
			}
		}
	}
	return types.Txxx, types.Txxx
}

func soleComponent(init *ir.Nodes, n ir.Node) ir.Node {
	if n.Type().SoleComponent() == nil {
		return n
	}
	// Keep in sync with cmd/compile/internal/types/type.go:Type.SoleComponent.
	for {
		switch {
		case n.Type().IsStruct():
			if n.Type().Field(0).Sym.IsBlank() {
				// Treat blank fields as the zero value as the Go language requires.
				n = typecheck.TempAt(base.Pos, ir.CurFunc, n.Type().Field(0).Type)
				appendWalkStmt(init, ir.NewAssignStmt(base.Pos, n, nil))
				continue
			}
			n = typecheck.DotField(n.Pos(), n, 0)
		case n.Type().IsArray():
			n = typecheck.Expr(ir.NewIndexExpr(n.Pos(), n, ir.NewInt(base.Pos, 0)))
		default:
			return n
		}
	}
}

// byteindex converts n, which is byte-sized, to an int used to index into an array.
// We cannot use conv, because we allow converting bool to int here,
// which is forbidden in user code.
func byteindex(n ir.Node) ir.Node {
	// We cannot convert from bool to int directly.
	// While converting from int8 to int is possible, it would yield
	// the wrong result for negative values.
	// Reinterpreting the value as an unsigned byte solves both cases.
	if !types.Identical(n.Type(), types.Types[types.TUINT8]) {
		n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
		n.SetType(types.Types[types.TUINT8])
		n.SetTypecheck(1)
	}
	n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
	n.SetType(types.Types[types.TINT])
	n.SetTypecheck(1)
	return n
}

func walkCheckPtrArithmetic(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	// Calling cheapExpr(n, init) below leads to a recursive call to
	// walkExpr, which leads us back here again. Use n.Checkptr to
	// prevent infinite loops.
	if n.CheckPtr() {
		return n
	}
	n.SetCheckPtr(true)
	defer n.SetCheckPtr(false)

	// TODO(mdempsky): Make stricter. We only need to exempt
	// reflect.Value.Pointer and reflect.Value.UnsafeAddr.
	switch n.X.Op() {
	case ir.OCALLMETH:
		base.FatalfAt(n.X.Pos(), "OCALLMETH missed by typecheck")
	case ir.OCALLFUNC, ir.OCALLINTER:
		return n
	}

	if n.X.Op() == ir.ODOTPTR && ir.IsReflectHeaderDataField(n.X) {
		return n
	}

	// Find original unsafe.Pointer operands involved in this
	// arithmetic expression.
	//
	// "It is valid both to add and to subtract offsets from a
	// pointer in this way. It is also valid to use &^ to round
	// pointers, usually for alignment."
	var originals []ir.Node
	var walk func(n ir.Node)
	walk = func(n ir.Node) {
		switch n.Op() {
		case ir.OADD:
			n := n.(*ir.BinaryExpr)
			walk(n.X)
			walk(n.Y)
		case ir.OSUB, ir.OANDNOT:
			n := n.(*ir.BinaryExpr)
			walk(n.X)
		case ir.OCONVNOP:
			n := n.(*ir.ConvExpr)
			if n.X.Type().IsUnsafePtr() {
				n.X = cheapExpr(n.X, init)
				originals = append(originals, typecheck.ConvNop(n.X, types.Types[types.TUNSAFEPTR]))
			}
		}
	}
	walk(n.X)

	cheap := cheapExpr(n, init)

	slice := typecheck.MakeDotArgs(base.Pos, types.NewSlice(types.Types[types.TUNSAFEPTR]), originals)
	slice.SetEsc(ir.EscNone)

	init.Append(mkcall("checkptrArithmetic", nil, init, typecheck.ConvNop(cheap, types.Types[types.TUNSAFEPTR]), slice))
	// TODO(khr): Mark backing store of slice as dead. This will allow us to reuse
	// the backing store for multiple calls to checkptrArithmetic.

	return cheap
}

// walkSliceToArray walks an OSLICE2ARR expression.
func walkSliceToArray(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
	// Replace T(x) with *(*T)(x).
	conv := typecheck.Expr(ir.NewConvExpr(base.Pos, ir.OCONV, types.NewPtr(n.Type()), n.X)).(*ir.ConvExpr)
	deref := typecheck.Expr(ir.NewStarExpr(base.Pos, conv)).(*ir.StarExpr)

	// The OSLICE2ARRPTR conversion handles checking the slice length,
	// so the dereference can't fail.
	//
	// However, this is more than just an optimization: if T is a
	// zero-length array, then x (and thus (*T)(x)) can be nil, but T(x)
	// should *not* panic. So suppressing the nil check here is
	// necessary for correctness in that case.
	deref.SetBounded(true)

	return walkExpr(deref, init)
}
