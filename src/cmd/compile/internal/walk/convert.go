// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"encoding/binary"
	"go/constant"

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
		if n.Type().IsPtr() && n.X.Type().IsUnsafePtr() { // unsafe.Pointer to *T
			return walkCheckPtrAlignment(n, init, nil)
		}
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

	if !fromType.IsInterface() && !ir.IsBlank(ir.CurFunc.Nname) { // skip unnamed functions (func _())
		reflectdata.MarkTypeUsedInInterface(fromType, ir.CurFunc.LSym)
	}

	// typeword generates the type word of the interface value.
	typeword := func() ir.Node {
		if toType.IsEmptyInterface() {
			return reflectdata.TypePtr(fromType)
		}
		return reflectdata.ITabAddr(fromType, toType)
	}

	// Optimize convT2E or convT2I as a two-word copy when T is pointer-shaped.
	if types.IsDirectIface(fromType) {
		l := ir.NewBinaryExpr(base.Pos, ir.OEFACE, typeword(), n.X)
		l.SetType(toType)
		l.SetTypecheck(n.Typecheck())
		return l
	}

	// Optimize convT2{E,I} for many cases in which T is not pointer-shaped,
	// by using an existing addressable value identical to n.Left
	// or creating one on the stack.
	var value ir.Node
	switch {
	case fromType.Size() == 0:
		// n.Left is zero-sized. Use zerobase.
		cheapExpr(n.X, init) // Evaluate n.Left for side-effects. See issue 19246.
		value = ir.NewLinksymExpr(base.Pos, ir.Syms.Zerobase, types.Types[types.TUINTPTR])
	case fromType.IsBoolean() || (fromType.Size() == 1 && fromType.IsInteger()):
		// n.Left is a bool/byte. Use staticuint64s[n.Left * 8] on little-endian
		// and staticuint64s[n.Left * 8 + 7] on big-endian.
		n.X = cheapExpr(n.X, init)
		// byteindex widens n.Left so that the multiplication doesn't overflow.
		index := ir.NewBinaryExpr(base.Pos, ir.OLSH, byteindex(n.X), ir.NewInt(3))
		if ssagen.Arch.LinkArch.ByteOrder == binary.BigEndian {
			index = ir.NewBinaryExpr(base.Pos, ir.OADD, index, ir.NewInt(7))
		}
		// The actual type is [256]uint64, but we use [256*8]uint8 so we can address
		// individual bytes.
		staticuint64s := ir.NewLinksymExpr(base.Pos, ir.Syms.Staticuint64s, types.NewArray(types.Types[types.TUINT8], 256*8))
		xe := ir.NewIndexExpr(base.Pos, staticuint64s, index)
		xe.SetBounded(true)
		value = xe
	case n.X.Op() == ir.ONAME && n.X.(*ir.Name).Class == ir.PEXTERN && n.X.(*ir.Name).Readonly():
		// n.Left is a readonly global; use it directly.
		value = n.X
	case !fromType.IsInterface() && n.Esc() == ir.EscNone && fromType.Width <= 1024:
		// n.Left does not escape. Use a stack temporary initialized to n.Left.
		value = typecheck.Temp(fromType)
		init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, value, n.X)))
	}

	if value != nil {
		// Value is identical to n.Left.
		// Construct the interface directly: {type/itab, &value}.
		l := ir.NewBinaryExpr(base.Pos, ir.OEFACE, typeword(), typecheck.Expr(typecheck.NodAddr(value)))
		l.SetType(toType)
		l.SetTypecheck(n.Typecheck())
		return l
	}

	// Implement interface to empty interface conversion.
	// tmp = i.itab
	// if tmp != nil {
	//    tmp = tmp.type
	// }
	// e = iface{tmp, i.data}
	if toType.IsEmptyInterface() && fromType.IsInterface() && !fromType.IsEmptyInterface() {
		// Evaluate the input interface.
		c := typecheck.Temp(fromType)
		init.Append(ir.NewAssignStmt(base.Pos, c, n.X))

		// Get the itab out of the interface.
		tmp := typecheck.Temp(types.NewPtr(types.Types[types.TUINT8]))
		init.Append(ir.NewAssignStmt(base.Pos, tmp, typecheck.Expr(ir.NewUnaryExpr(base.Pos, ir.OITAB, c))))

		// Get the type out of the itab.
		nif := ir.NewIfStmt(base.Pos, typecheck.Expr(ir.NewBinaryExpr(base.Pos, ir.ONE, tmp, typecheck.NodNil())), nil, nil)
		nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, tmp, itabType(tmp))}
		init.Append(nif)

		// Build the result.
		e := ir.NewBinaryExpr(base.Pos, ir.OEFACE, tmp, ifaceData(n.Pos(), c, types.NewPtr(types.Types[types.TUINT8])))
		e.SetType(toType) // assign type manually, typecheck doesn't understand OEFACE.
		e.SetTypecheck(1)
		return e
	}

	fnname, argType, needsaddr := convFuncName(fromType, toType)

	if !needsaddr && !fromType.IsInterface() {
		// Use a specialized conversion routine that only returns a data pointer.
		// ptr = convT2X(val)
		// e = iface{typ/tab, ptr}
		fn := typecheck.LookupRuntime(fnname)
		types.CalcSize(fromType)

		arg := n.X
		switch {
		case fromType == argType:
			// already in the right type, nothing to do
		case fromType.Kind() == argType.Kind(),
			fromType.IsPtrShaped() && argType.IsPtrShaped():
			// can directly convert (e.g. named type to underlying type, or one pointer to another)
			arg = ir.NewConvExpr(n.Pos(), ir.OCONVNOP, argType, arg)
		case fromType.IsInteger() && argType.IsInteger():
			// can directly convert (e.g. int32 to uint32)
			arg = ir.NewConvExpr(n.Pos(), ir.OCONV, argType, arg)
		default:
			// unsafe cast through memory
			arg = copyExpr(arg, arg.Type(), init)
			var addr ir.Node = typecheck.NodAddr(arg)
			addr = ir.NewConvExpr(n.Pos(), ir.OCONVNOP, argType.PtrTo(), addr)
			arg = ir.NewStarExpr(n.Pos(), addr)
			arg.SetType(argType)
		}

		call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, nil)
		call.Args = []ir.Node{arg}
		e := ir.NewBinaryExpr(base.Pos, ir.OEFACE, typeword(), safeExpr(walkExpr(typecheck.Expr(call), init), init))
		e.SetType(toType)
		e.SetTypecheck(1)
		return e
	}

	var tab ir.Node
	if fromType.IsInterface() {
		// convI2I
		tab = reflectdata.TypePtr(toType)
	} else {
		// convT2x
		tab = typeword()
	}

	v := n.X
	if needsaddr {
		// Types of large or unknown size are passed by reference.
		// Orderexpr arranged for n.Left to be a temporary for all
		// the conversions it could see. Comparison of an interface
		// with a non-interface, especially in a switch on interface value
		// with non-interface cases, is not visible to order.stmt, so we
		// have to fall back on allocating a temp here.
		if !ir.IsAddressable(v) {
			v = copyExpr(v, v.Type(), init)
		}
		v = typecheck.NodAddr(v)
	}

	types.CalcSize(fromType)
	fn := typecheck.LookupRuntime(fnname)
	fn = typecheck.SubstArgTypes(fn, fromType, toType)
	types.CalcSize(fn.Type())
	call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, nil)
	call.Args = []ir.Node{tab, v}
	return walkExpr(typecheck.Expr(call), init)
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
		p := typecheck.Temp(t.PtrTo()) // *[n]byte
		init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, p, a)))

		// Copy from the static string data to the [n]byte.
		if len(sc) > 0 {
			as := ir.NewAssignStmt(base.Pos, ir.NewStarExpr(base.Pos, p), ir.NewStarExpr(base.Pos, typecheck.ConvNop(ir.NewUnaryExpr(base.Pos, ir.OSPTR, s), t.PtrTo())))
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

// convFuncName builds the runtime function name for interface conversion.
// It also returns the argument type that the runtime function takes, and
// whether the function expects the data by address.
// Not all names are possible. For example, we never generate convE2E or convE2I.
func convFuncName(from, to *types.Type) (fnname string, argType *types.Type, needsaddr bool) {
	tkind := to.Tie()
	switch from.Tie() {
	case 'I':
		if tkind == 'I' {
			return "convI2I", types.Types[types.TINTER], false
		}
	case 'T':
		switch {
		case from.Size() == 2 && from.Align == 2:
			return "convT16", types.Types[types.TUINT16], false
		case from.Size() == 4 && from.Align == 4 && !from.HasPointers():
			return "convT32", types.Types[types.TUINT32], false
		case from.Size() == 8 && from.Align == types.Types[types.TUINT64].Align && !from.HasPointers():
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

		switch tkind {
		case 'E':
			if !from.HasPointers() {
				return "convT2Enoptr", types.Types[types.TUNSAFEPTR], true
			}
			return "convT2E", types.Types[types.TUNSAFEPTR], true
		case 'I':
			if !from.HasPointers() {
				return "convT2Inoptr", types.Types[types.TUNSAFEPTR], true
			}
			return "convT2I", types.Types[types.TUNSAFEPTR], true
		}
	}
	base.Fatalf("unknown conv func %c2%c", from.Tie(), to.Tie())
	panic("unreachable")
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
				return src.Kind(), types.TFLOAT64
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
				return src.Kind(), types.TFLOAT64
			case types.TUINT32, types.TUINT, types.TUINTPTR:
				return types.TUINT32, types.TFLOAT64
			}
		}
	}
	return types.Txxx, types.Txxx
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

func walkCheckPtrAlignment(n *ir.ConvExpr, init *ir.Nodes, count ir.Node) ir.Node {
	if !n.Type().IsPtr() {
		base.Fatalf("expected pointer type: %v", n.Type())
	}
	elem := n.Type().Elem()
	if count != nil {
		if !elem.IsArray() {
			base.Fatalf("expected array type: %v", elem)
		}
		elem = elem.Elem()
	}

	size := elem.Size()
	if elem.Alignment() == 1 && (size == 0 || size == 1 && count == nil) {
		return n
	}

	if count == nil {
		count = ir.NewInt(1)
	}

	n.X = cheapExpr(n.X, init)
	init.Append(mkcall("checkptrAlignment", nil, init, typecheck.ConvNop(n.X, types.Types[types.TUNSAFEPTR]), reflectdata.TypePtr(elem), typecheck.Conv(count, types.Types[types.TUINTPTR])))
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
	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
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

	slice := typecheck.MakeDotArgs(types.NewSlice(types.Types[types.TUNSAFEPTR]), originals)
	slice.SetEsc(ir.EscNone)

	init.Append(mkcall("checkptrArithmetic", nil, init, typecheck.ConvNop(cheap, types.Types[types.TUNSAFEPTR]), slice))
	// TODO(khr): Mark backing store of slice as dead. This will allow us to reuse
	// the backing store for multiple calls to checkptrArithmetic.

	return cheap
}
