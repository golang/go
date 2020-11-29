// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"go/constant"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// largeStack is info about a function whose stack frame is too large (rare).
type largeStack struct {
	locals int64
	args   int64
	callee int64
	pos    src.XPos
}

var (
	largeStackFramesMu sync.Mutex // protects largeStackFrames
	largeStackFrames   []largeStack
)

// hasUniquePos reports whether n has a unique position that can be
// used for reporting error messages.
//
// It's primarily used to distinguish references to named objects,
// whose Pos will point back to their declaration position rather than
// their usage position.
func hasUniquePos(n ir.Node) bool {
	switch n.Op() {
	case ir.ONAME, ir.OPACK:
		return false
	case ir.OLITERAL, ir.ONIL, ir.OTYPE:
		if n.Sym() != nil {
			return false
		}
	}

	if !n.Pos().IsKnown() {
		if base.Flag.K != 0 {
			base.Warn("setlineno: unknown position (line 0)")
		}
		return false
	}

	return true
}

func setlineno(n ir.Node) src.XPos {
	lno := base.Pos
	if n != nil && hasUniquePos(n) {
		base.Pos = n.Pos()
	}
	return lno
}

func lookup(name string) *types.Sym {
	return ir.LocalPkg.Lookup(name)
}

// lookupN looks up the symbol starting with prefix and ending with
// the decimal n. If prefix is too long, lookupN panics.
func lookupN(prefix string, n int) *types.Sym {
	var buf [20]byte // plenty long enough for all current users
	copy(buf[:], prefix)
	b := strconv.AppendInt(buf[:len(prefix)], int64(n), 10)
	return ir.LocalPkg.LookupBytes(b)
}

// autolabel generates a new Name node for use with
// an automatically generated label.
// prefix is a short mnemonic (e.g. ".s" for switch)
// to help with debugging.
// It should begin with "." to avoid conflicts with
// user labels.
func autolabel(prefix string) *types.Sym {
	if prefix[0] != '.' {
		base.Fatalf("autolabel prefix must start with '.', have %q", prefix)
	}
	fn := Curfn
	if Curfn == nil {
		base.Fatalf("autolabel outside function")
	}
	n := fn.Label
	fn.Label++
	return lookupN(prefix, int(n))
}

// find all the exported symbols in package opkg
// and make them available in the current package
func importdot(opkg *types.Pkg, pack *ir.PkgName) {
	n := 0
	for _, s := range opkg.Syms {
		if s.Def == nil {
			continue
		}
		if !types.IsExported(s.Name) || strings.ContainsRune(s.Name, 0xb7) { // 0xb7 = center dot
			continue
		}
		s1 := lookup(s.Name)
		if s1.Def != nil {
			pkgerror := fmt.Sprintf("during import %q", opkg.Path)
			redeclare(base.Pos, s1, pkgerror)
			continue
		}

		s1.Def = s.Def
		s1.Block = s.Block
		if ir.AsNode(s1.Def).Name() == nil {
			ir.Dump("s1def", ir.AsNode(s1.Def))
			base.Fatalf("missing Name")
		}
		ir.AsNode(s1.Def).Name().PkgName = pack
		s1.Origpkg = opkg
		n++
	}

	if n == 0 {
		// can't possibly be used - there were no symbols
		base.ErrorfAt(pack.Pos(), "imported and not used: %q", opkg.Path)
	}
}

// newname returns a new ONAME Node associated with symbol s.
func NewName(s *types.Sym) *ir.Name {
	n := ir.NewNameAt(base.Pos, s)
	n.Curfn = Curfn
	return n
}

// nodSym makes a Node with Op op and with the Left field set to left
// and the Sym field set to sym. This is for ODOT and friends.
func nodSym(op ir.Op, left ir.Node, sym *types.Sym) ir.Node {
	return nodlSym(base.Pos, op, left, sym)
}

// nodlSym makes a Node with position Pos, with Op op, and with the Left field set to left
// and the Sym field set to sym. This is for ODOT and friends.
func nodlSym(pos src.XPos, op ir.Op, left ir.Node, sym *types.Sym) ir.Node {
	n := ir.NodAt(pos, op, left, nil)
	n.SetSym(sym)
	return n
}

// methcmp sorts methods by symbol.
type methcmp []*types.Field

func (x methcmp) Len() int           { return len(x) }
func (x methcmp) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x methcmp) Less(i, j int) bool { return x[i].Sym.Less(x[j].Sym) }

func nodintconst(v int64) ir.Node {
	return ir.NewLiteral(constant.MakeInt64(v))
}

func nodnil() ir.Node {
	n := ir.Nod(ir.ONIL, nil, nil)
	n.SetType(types.Types[types.TNIL])
	return n
}

func nodbool(b bool) ir.Node {
	return ir.NewLiteral(constant.MakeBool(b))
}

func nodstr(s string) ir.Node {
	return ir.NewLiteral(constant.MakeString(s))
}

func isptrto(t *types.Type, et types.EType) bool {
	if t == nil {
		return false
	}
	if !t.IsPtr() {
		return false
	}
	t = t.Elem()
	if t == nil {
		return false
	}
	if t.Etype != et {
		return false
	}
	return true
}

// methtype returns the underlying type, if any,
// that owns methods with receiver parameter t.
// The result is either a named type or an anonymous struct.
func methtype(t *types.Type) *types.Type {
	if t == nil {
		return nil
	}

	// Strip away pointer if it's there.
	if t.IsPtr() {
		if t.Sym != nil {
			return nil
		}
		t = t.Elem()
		if t == nil {
			return nil
		}
	}

	// Must be a named type or anonymous struct.
	if t.Sym == nil && !t.IsStruct() {
		return nil
	}

	// Check types.
	if issimple[t.Etype] {
		return t
	}
	switch t.Etype {
	case types.TARRAY, types.TCHAN, types.TFUNC, types.TMAP, types.TSLICE, types.TSTRING, types.TSTRUCT:
		return t
	}
	return nil
}

// Is type src assignment compatible to type dst?
// If so, return op code to use in conversion.
// If not, return OXXX. In this case, the string return parameter may
// hold a reason why. In all other cases, it'll be the empty string.
func assignop(src, dst *types.Type) (ir.Op, string) {
	if src == dst {
		return ir.OCONVNOP, ""
	}
	if src == nil || dst == nil || src.Etype == types.TFORW || dst.Etype == types.TFORW || src.Orig == nil || dst.Orig == nil {
		return ir.OXXX, ""
	}

	// 1. src type is identical to dst.
	if types.Identical(src, dst) {
		return ir.OCONVNOP, ""
	}

	// 2. src and dst have identical underlying types
	// and either src or dst is not a named type or
	// both are empty interface types.
	// For assignable but different non-empty interface types,
	// we want to recompute the itab. Recomputing the itab ensures
	// that itabs are unique (thus an interface with a compile-time
	// type I has an itab with interface type I).
	if types.Identical(src.Orig, dst.Orig) {
		if src.IsEmptyInterface() {
			// Conversion between two empty interfaces
			// requires no code.
			return ir.OCONVNOP, ""
		}
		if (src.Sym == nil || dst.Sym == nil) && !src.IsInterface() {
			// Conversion between two types, at least one unnamed,
			// needs no conversion. The exception is nonempty interfaces
			// which need to have their itab updated.
			return ir.OCONVNOP, ""
		}
	}

	// 3. dst is an interface type and src implements dst.
	if dst.IsInterface() && src.Etype != types.TNIL {
		var missing, have *types.Field
		var ptr int
		if implements(src, dst, &missing, &have, &ptr) {
			return ir.OCONVIFACE, ""
		}

		// we'll have complained about this method anyway, suppress spurious messages.
		if have != nil && have.Sym == missing.Sym && (have.Type.Broke() || missing.Type.Broke()) {
			return ir.OCONVIFACE, ""
		}

		var why string
		if isptrto(src, types.TINTER) {
			why = fmt.Sprintf(":\n\t%v is pointer to interface, not interface", src)
		} else if have != nil && have.Sym == missing.Sym && have.Nointerface() {
			why = fmt.Sprintf(":\n\t%v does not implement %v (%v method is marked 'nointerface')", src, dst, missing.Sym)
		} else if have != nil && have.Sym == missing.Sym {
			why = fmt.Sprintf(":\n\t%v does not implement %v (wrong type for %v method)\n"+
				"\t\thave %v%0S\n\t\twant %v%0S", src, dst, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
		} else if ptr != 0 {
			why = fmt.Sprintf(":\n\t%v does not implement %v (%v method has pointer receiver)", src, dst, missing.Sym)
		} else if have != nil {
			why = fmt.Sprintf(":\n\t%v does not implement %v (missing %v method)\n"+
				"\t\thave %v%0S\n\t\twant %v%0S", src, dst, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
		} else {
			why = fmt.Sprintf(":\n\t%v does not implement %v (missing %v method)", src, dst, missing.Sym)
		}

		return ir.OXXX, why
	}

	if isptrto(dst, types.TINTER) {
		why := fmt.Sprintf(":\n\t%v is pointer to interface, not interface", dst)
		return ir.OXXX, why
	}

	if src.IsInterface() && dst.Etype != types.TBLANK {
		var missing, have *types.Field
		var ptr int
		var why string
		if implements(dst, src, &missing, &have, &ptr) {
			why = ": need type assertion"
		}
		return ir.OXXX, why
	}

	// 4. src is a bidirectional channel value, dst is a channel type,
	// src and dst have identical element types, and
	// either src or dst is not a named type.
	if src.IsChan() && src.ChanDir() == types.Cboth && dst.IsChan() {
		if types.Identical(src.Elem(), dst.Elem()) && (src.Sym == nil || dst.Sym == nil) {
			return ir.OCONVNOP, ""
		}
	}

	// 5. src is the predeclared identifier nil and dst is a nillable type.
	if src.Etype == types.TNIL {
		switch dst.Etype {
		case types.TPTR,
			types.TFUNC,
			types.TMAP,
			types.TCHAN,
			types.TINTER,
			types.TSLICE:
			return ir.OCONVNOP, ""
		}
	}

	// 6. rule about untyped constants - already converted by defaultlit.

	// 7. Any typed value can be assigned to the blank identifier.
	if dst.Etype == types.TBLANK {
		return ir.OCONVNOP, ""
	}

	return ir.OXXX, ""
}

// Can we convert a value of type src to a value of type dst?
// If so, return op code to use in conversion (maybe OCONVNOP).
// If not, return OXXX. In this case, the string return parameter may
// hold a reason why. In all other cases, it'll be the empty string.
// srcConstant indicates whether the value of type src is a constant.
func convertop(srcConstant bool, src, dst *types.Type) (ir.Op, string) {
	if src == dst {
		return ir.OCONVNOP, ""
	}
	if src == nil || dst == nil {
		return ir.OXXX, ""
	}

	// Conversions from regular to go:notinheap are not allowed
	// (unless it's unsafe.Pointer). These are runtime-specific
	// rules.
	// (a) Disallow (*T) to (*U) where T is go:notinheap but U isn't.
	if src.IsPtr() && dst.IsPtr() && dst.Elem().NotInHeap() && !src.Elem().NotInHeap() {
		why := fmt.Sprintf(":\n\t%v is incomplete (or unallocatable), but %v is not", dst.Elem(), src.Elem())
		return ir.OXXX, why
	}
	// (b) Disallow string to []T where T is go:notinheap.
	if src.IsString() && dst.IsSlice() && dst.Elem().NotInHeap() && (dst.Elem().Etype == types.Bytetype.Etype || dst.Elem().Etype == types.Runetype.Etype) {
		why := fmt.Sprintf(":\n\t%v is incomplete (or unallocatable)", dst.Elem())
		return ir.OXXX, why
	}

	// 1. src can be assigned to dst.
	op, why := assignop(src, dst)
	if op != ir.OXXX {
		return op, why
	}

	// The rules for interfaces are no different in conversions
	// than assignments. If interfaces are involved, stop now
	// with the good message from assignop.
	// Otherwise clear the error.
	if src.IsInterface() || dst.IsInterface() {
		return ir.OXXX, why
	}

	// 2. Ignoring struct tags, src and dst have identical underlying types.
	if types.IdenticalIgnoreTags(src.Orig, dst.Orig) {
		return ir.OCONVNOP, ""
	}

	// 3. src and dst are unnamed pointer types and, ignoring struct tags,
	// their base types have identical underlying types.
	if src.IsPtr() && dst.IsPtr() && src.Sym == nil && dst.Sym == nil {
		if types.IdenticalIgnoreTags(src.Elem().Orig, dst.Elem().Orig) {
			return ir.OCONVNOP, ""
		}
	}

	// 4. src and dst are both integer or floating point types.
	if (src.IsInteger() || src.IsFloat()) && (dst.IsInteger() || dst.IsFloat()) {
		if simtype[src.Etype] == simtype[dst.Etype] {
			return ir.OCONVNOP, ""
		}
		return ir.OCONV, ""
	}

	// 5. src and dst are both complex types.
	if src.IsComplex() && dst.IsComplex() {
		if simtype[src.Etype] == simtype[dst.Etype] {
			return ir.OCONVNOP, ""
		}
		return ir.OCONV, ""
	}

	// Special case for constant conversions: any numeric
	// conversion is potentially okay. We'll validate further
	// within evconst. See #38117.
	if srcConstant && (src.IsInteger() || src.IsFloat() || src.IsComplex()) && (dst.IsInteger() || dst.IsFloat() || dst.IsComplex()) {
		return ir.OCONV, ""
	}

	// 6. src is an integer or has type []byte or []rune
	// and dst is a string type.
	if src.IsInteger() && dst.IsString() {
		return ir.ORUNESTR, ""
	}

	if src.IsSlice() && dst.IsString() {
		if src.Elem().Etype == types.Bytetype.Etype {
			return ir.OBYTES2STR, ""
		}
		if src.Elem().Etype == types.Runetype.Etype {
			return ir.ORUNES2STR, ""
		}
	}

	// 7. src is a string and dst is []byte or []rune.
	// String to slice.
	if src.IsString() && dst.IsSlice() {
		if dst.Elem().Etype == types.Bytetype.Etype {
			return ir.OSTR2BYTES, ""
		}
		if dst.Elem().Etype == types.Runetype.Etype {
			return ir.OSTR2RUNES, ""
		}
	}

	// 8. src is a pointer or uintptr and dst is unsafe.Pointer.
	if (src.IsPtr() || src.IsUintptr()) && dst.IsUnsafePtr() {
		return ir.OCONVNOP, ""
	}

	// 9. src is unsafe.Pointer and dst is a pointer or uintptr.
	if src.IsUnsafePtr() && (dst.IsPtr() || dst.IsUintptr()) {
		return ir.OCONVNOP, ""
	}

	// src is map and dst is a pointer to corresponding hmap.
	// This rule is needed for the implementation detail that
	// go gc maps are implemented as a pointer to a hmap struct.
	if src.Etype == types.TMAP && dst.IsPtr() &&
		src.MapType().Hmap == dst.Elem() {
		return ir.OCONVNOP, ""
	}

	return ir.OXXX, ""
}

func assignconv(n ir.Node, t *types.Type, context string) ir.Node {
	return assignconvfn(n, t, func() string { return context })
}

// Convert node n for assignment to type t.
func assignconvfn(n ir.Node, t *types.Type, context func() string) ir.Node {
	if n == nil || n.Type() == nil || n.Type().Broke() {
		return n
	}

	if t.Etype == types.TBLANK && n.Type().Etype == types.TNIL {
		base.Errorf("use of untyped nil")
	}

	n = convlit1(n, t, false, context)
	if n.Type() == nil {
		return n
	}
	if t.Etype == types.TBLANK {
		return n
	}

	// Convert ideal bool from comparison to plain bool
	// if the next step is non-bool (like interface{}).
	if n.Type() == types.UntypedBool && !t.IsBoolean() {
		if n.Op() == ir.ONAME || n.Op() == ir.OLITERAL {
			r := ir.Nod(ir.OCONVNOP, n, nil)
			r.SetType(types.Types[types.TBOOL])
			r.SetTypecheck(1)
			r.SetImplicit(true)
			n = r
		}
	}

	if types.Identical(n.Type(), t) {
		return n
	}

	op, why := assignop(n.Type(), t)
	if op == ir.OXXX {
		base.Errorf("cannot use %L as type %v in %s%s", n, t, context(), why)
		op = ir.OCONV
	}

	r := ir.Nod(op, n, nil)
	r.SetType(t)
	r.SetTypecheck(1)
	r.SetImplicit(true)
	r.(ir.OrigNode).SetOrig(ir.Orig(n))
	return r
}

// backingArrayPtrLen extracts the pointer and length from a slice or string.
// This constructs two nodes referring to n, so n must be a cheapexpr.
func backingArrayPtrLen(n ir.Node) (ptr, len ir.Node) {
	var init ir.Nodes
	c := cheapexpr(n, &init)
	if c != n || init.Len() != 0 {
		base.Fatalf("backingArrayPtrLen not cheap: %v", n)
	}
	ptr = ir.Nod(ir.OSPTR, n, nil)
	if n.Type().IsString() {
		ptr.SetType(types.Types[types.TUINT8].PtrTo())
	} else {
		ptr.SetType(n.Type().Elem().PtrTo())
	}
	len = ir.Nod(ir.OLEN, n, nil)
	len.SetType(types.Types[types.TINT])
	return ptr, len
}

func syslook(name string) ir.Node {
	s := Runtimepkg.Lookup(name)
	if s == nil || s.Def == nil {
		base.Fatalf("syslook: can't find runtime.%s", name)
	}
	return ir.AsNode(s.Def)
}

// typehash computes a hash value for type t to use in type switch statements.
func typehash(t *types.Type) uint32 {
	p := t.LongString()

	// Using MD5 is overkill, but reduces accidental collisions.
	h := md5.Sum([]byte(p))
	return binary.LittleEndian.Uint32(h[:4])
}

// updateHasCall checks whether expression n contains any function
// calls and sets the n.HasCall flag if so.
func updateHasCall(n ir.Node) {
	if n == nil {
		return
	}
	n.SetHasCall(calcHasCall(n))
}

func calcHasCall(n ir.Node) bool {
	if n.Init().Len() != 0 {
		// TODO(mdempsky): This seems overly conservative.
		return true
	}

	switch n.Op() {
	case ir.OLITERAL, ir.ONIL, ir.ONAME, ir.OTYPE:
		if n.HasCall() {
			base.Fatalf("OLITERAL/ONAME/OTYPE should never have calls: %+v", n)
		}
		return false
	case ir.OCALL, ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		return true
	case ir.OANDAND, ir.OOROR:
		// hard with instrumented code
		if instrumenting {
			return true
		}
	case ir.OINDEX, ir.OSLICE, ir.OSLICEARR, ir.OSLICE3, ir.OSLICE3ARR, ir.OSLICESTR,
		ir.ODEREF, ir.ODOTPTR, ir.ODOTTYPE, ir.ODIV, ir.OMOD:
		// These ops might panic, make sure they are done
		// before we start marshaling args for a call. See issue 16760.
		return true

	// When using soft-float, these ops might be rewritten to function calls
	// so we ensure they are evaluated first.
	case ir.OADD, ir.OSUB, ir.ONEG, ir.OMUL:
		if thearch.SoftFloat && (isFloat[n.Type().Etype] || isComplex[n.Type().Etype]) {
			return true
		}
	case ir.OLT, ir.OEQ, ir.ONE, ir.OLE, ir.OGE, ir.OGT:
		if thearch.SoftFloat && (isFloat[n.Left().Type().Etype] || isComplex[n.Left().Type().Etype]) {
			return true
		}
	case ir.OCONV:
		if thearch.SoftFloat && ((isFloat[n.Type().Etype] || isComplex[n.Type().Etype]) || (isFloat[n.Left().Type().Etype] || isComplex[n.Left().Type().Etype])) {
			return true
		}
	}

	if n.Left() != nil && n.Left().HasCall() {
		return true
	}
	if n.Right() != nil && n.Right().HasCall() {
		return true
	}
	return false
}

func badtype(op ir.Op, tl, tr *types.Type) {
	var s string
	if tl != nil {
		s += fmt.Sprintf("\n\t%v", tl)
	}
	if tr != nil {
		s += fmt.Sprintf("\n\t%v", tr)
	}

	// common mistake: *struct and *interface.
	if tl != nil && tr != nil && tl.IsPtr() && tr.IsPtr() {
		if tl.Elem().IsStruct() && tr.Elem().IsInterface() {
			s += "\n\t(*struct vs *interface)"
		} else if tl.Elem().IsInterface() && tr.Elem().IsStruct() {
			s += "\n\t(*interface vs *struct)"
		}
	}

	base.Errorf("illegal types for operand: %v%s", op, s)
}

// brcom returns !(op).
// For example, brcom(==) is !=.
func brcom(op ir.Op) ir.Op {
	switch op {
	case ir.OEQ:
		return ir.ONE
	case ir.ONE:
		return ir.OEQ
	case ir.OLT:
		return ir.OGE
	case ir.OGT:
		return ir.OLE
	case ir.OLE:
		return ir.OGT
	case ir.OGE:
		return ir.OLT
	}
	base.Fatalf("brcom: no com for %v\n", op)
	return op
}

// brrev returns reverse(op).
// For example, Brrev(<) is >.
func brrev(op ir.Op) ir.Op {
	switch op {
	case ir.OEQ:
		return ir.OEQ
	case ir.ONE:
		return ir.ONE
	case ir.OLT:
		return ir.OGT
	case ir.OGT:
		return ir.OLT
	case ir.OLE:
		return ir.OGE
	case ir.OGE:
		return ir.OLE
	}
	base.Fatalf("brrev: no rev for %v\n", op)
	return op
}

// return side effect-free n, appending side effects to init.
// result is assignable if n is.
func safeexpr(n ir.Node, init *ir.Nodes) ir.Node {
	if n == nil {
		return nil
	}

	if n.Init().Len() != 0 {
		walkstmtlist(n.Init().Slice())
		init.AppendNodes(n.PtrInit())
	}

	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n

	case ir.ODOT, ir.OLEN, ir.OCAP:
		l := safeexpr(n.Left(), init)
		if l == n.Left() {
			return n
		}
		r := ir.Copy(n)
		r.SetLeft(l)
		r = typecheck(r, ctxExpr)
		r = walkexpr(r, init)
		return r

	case ir.ODOTPTR, ir.ODEREF:
		l := safeexpr(n.Left(), init)
		if l == n.Left() {
			return n
		}
		a := ir.Copy(n)
		a.SetLeft(l)
		a = walkexpr(a, init)
		return a

	case ir.OINDEX, ir.OINDEXMAP:
		l := safeexpr(n.Left(), init)
		r := safeexpr(n.Right(), init)
		if l == n.Left() && r == n.Right() {
			return n
		}
		a := ir.Copy(n)
		a.SetLeft(l)
		a.SetRight(r)
		a = walkexpr(a, init)
		return a

	case ir.OSTRUCTLIT, ir.OARRAYLIT, ir.OSLICELIT:
		if isStaticCompositeLiteral(n) {
			return n
		}
	}

	// make a copy; must not be used as an lvalue
	if islvalue(n) {
		base.Fatalf("missing lvalue case in safeexpr: %v", n)
	}
	return cheapexpr(n, init)
}

func copyexpr(n ir.Node, t *types.Type, init *ir.Nodes) ir.Node {
	l := temp(t)
	a := ir.Nod(ir.OAS, l, n)
	a = typecheck(a, ctxStmt)
	a = walkexpr(a, init)
	init.Append(a)
	return l
}

// return side-effect free and cheap n, appending side effects to init.
// result may not be assignable.
func cheapexpr(n ir.Node, init *ir.Nodes) ir.Node {
	switch n.Op() {
	case ir.ONAME, ir.OLITERAL, ir.ONIL:
		return n
	}

	return copyexpr(n, n.Type(), init)
}

// Code to resolve elided DOTs in embedded types.

// A Dlist stores a pointer to a TFIELD Type embedded within
// a TSTRUCT or TINTER Type.
type Dlist struct {
	field *types.Field
}

// dotlist is used by adddot1 to record the path of embedded fields
// used to access a target field or method.
// Must be non-nil so that dotpath returns a non-nil slice even if d is zero.
var dotlist = make([]Dlist, 10)

// lookdot0 returns the number of fields or methods named s associated
// with Type t. If exactly one exists, it will be returned in *save
// (if save is not nil).
func lookdot0(s *types.Sym, t *types.Type, save **types.Field, ignorecase bool) int {
	u := t
	if u.IsPtr() {
		u = u.Elem()
	}

	c := 0
	if u.IsStruct() || u.IsInterface() {
		for _, f := range u.Fields().Slice() {
			if f.Sym == s || (ignorecase && f.IsMethod() && strings.EqualFold(f.Sym.Name, s.Name)) {
				if save != nil {
					*save = f
				}
				c++
			}
		}
	}

	u = t
	if t.Sym != nil && t.IsPtr() && !t.Elem().IsPtr() {
		// If t is a defined pointer type, then x.m is shorthand for (*x).m.
		u = t.Elem()
	}
	u = methtype(u)
	if u != nil {
		for _, f := range u.Methods().Slice() {
			if f.Embedded == 0 && (f.Sym == s || (ignorecase && strings.EqualFold(f.Sym.Name, s.Name))) {
				if save != nil {
					*save = f
				}
				c++
			}
		}
	}

	return c
}

// adddot1 returns the number of fields or methods named s at depth d in Type t.
// If exactly one exists, it will be returned in *save (if save is not nil),
// and dotlist will contain the path of embedded fields traversed to find it,
// in reverse order. If none exist, more will indicate whether t contains any
// embedded fields at depth d, so callers can decide whether to retry at
// a greater depth.
func adddot1(s *types.Sym, t *types.Type, d int, save **types.Field, ignorecase bool) (c int, more bool) {
	if t.Recur() {
		return
	}
	t.SetRecur(true)
	defer t.SetRecur(false)

	var u *types.Type
	d--
	if d < 0 {
		// We've reached our target depth. If t has any fields/methods
		// named s, then we're done. Otherwise, we still need to check
		// below for embedded fields.
		c = lookdot0(s, t, save, ignorecase)
		if c != 0 {
			return c, false
		}
	}

	u = t
	if u.IsPtr() {
		u = u.Elem()
	}
	if !u.IsStruct() && !u.IsInterface() {
		return c, false
	}

	for _, f := range u.Fields().Slice() {
		if f.Embedded == 0 || f.Sym == nil {
			continue
		}
		if d < 0 {
			// Found an embedded field at target depth.
			return c, true
		}
		a, more1 := adddot1(s, f.Type, d, save, ignorecase)
		if a != 0 && c == 0 {
			dotlist[d].field = f
		}
		c += a
		if more1 {
			more = true
		}
	}

	return c, more
}

// dotpath computes the unique shortest explicit selector path to fully qualify
// a selection expression x.f, where x is of type t and f is the symbol s.
// If no such path exists, dotpath returns nil.
// If there are multiple shortest paths to the same depth, ambig is true.
func dotpath(s *types.Sym, t *types.Type, save **types.Field, ignorecase bool) (path []Dlist, ambig bool) {
	// The embedding of types within structs imposes a tree structure onto
	// types: structs parent the types they embed, and types parent their
	// fields or methods. Our goal here is to find the shortest path to
	// a field or method named s in the subtree rooted at t. To accomplish
	// that, we iteratively perform depth-first searches of increasing depth
	// until we either find the named field/method or exhaust the tree.
	for d := 0; ; d++ {
		if d > len(dotlist) {
			dotlist = append(dotlist, Dlist{})
		}
		if c, more := adddot1(s, t, d, save, ignorecase); c == 1 {
			return dotlist[:d], false
		} else if c > 1 {
			return nil, true
		} else if !more {
			return nil, false
		}
	}
}

// in T.field
// find missing fields that
// will give shortest unique addressing.
// modify the tree with missing type names.
func adddot(n ir.Node) ir.Node {
	n.SetLeft(typecheck(n.Left(), ctxType|ctxExpr))
	if n.Left().Diag() {
		n.SetDiag(true)
	}
	t := n.Left().Type()
	if t == nil {
		return n
	}

	if n.Left().Op() == ir.OTYPE {
		return n
	}

	s := n.Sym()
	if s == nil {
		return n
	}

	switch path, ambig := dotpath(s, t, nil, false); {
	case path != nil:
		// rebuild elided dots
		for c := len(path) - 1; c >= 0; c-- {
			n.SetLeft(nodSym(ir.ODOT, n.Left(), path[c].field.Sym))
			n.Left().SetImplicit(true)
		}
	case ambig:
		base.Errorf("ambiguous selector %v", n)
		n.SetLeft(nil)
	}

	return n
}

// Code to help generate trampoline functions for methods on embedded
// types. These are approx the same as the corresponding adddot
// routines except that they expect to be called with unique tasks and
// they return the actual methods.

type Symlink struct {
	field *types.Field
}

var slist []Symlink

func expand0(t *types.Type) {
	u := t
	if u.IsPtr() {
		u = u.Elem()
	}

	if u.IsInterface() {
		for _, f := range u.Fields().Slice() {
			if f.Sym.Uniq() {
				continue
			}
			f.Sym.SetUniq(true)
			slist = append(slist, Symlink{field: f})
		}

		return
	}

	u = methtype(t)
	if u != nil {
		for _, f := range u.Methods().Slice() {
			if f.Sym.Uniq() {
				continue
			}
			f.Sym.SetUniq(true)
			slist = append(slist, Symlink{field: f})
		}
	}
}

func expand1(t *types.Type, top bool) {
	if t.Recur() {
		return
	}
	t.SetRecur(true)

	if !top {
		expand0(t)
	}

	u := t
	if u.IsPtr() {
		u = u.Elem()
	}

	if u.IsStruct() || u.IsInterface() {
		for _, f := range u.Fields().Slice() {
			if f.Embedded == 0 {
				continue
			}
			if f.Sym == nil {
				continue
			}
			expand1(f.Type, false)
		}
	}

	t.SetRecur(false)
}

func expandmeth(t *types.Type) {
	if t == nil || t.AllMethods().Len() != 0 {
		return
	}

	// mark top-level method symbols
	// so that expand1 doesn't consider them.
	for _, f := range t.Methods().Slice() {
		f.Sym.SetUniq(true)
	}

	// generate all reachable methods
	slist = slist[:0]
	expand1(t, true)

	// check each method to be uniquely reachable
	var ms []*types.Field
	for i, sl := range slist {
		slist[i].field = nil
		sl.field.Sym.SetUniq(false)

		var f *types.Field
		path, _ := dotpath(sl.field.Sym, t, &f, false)
		if path == nil {
			continue
		}

		// dotpath may have dug out arbitrary fields, we only want methods.
		if !f.IsMethod() {
			continue
		}

		// add it to the base type method list
		f = f.Copy()
		f.Embedded = 1 // needs a trampoline
		for _, d := range path {
			if d.field.Type.IsPtr() {
				f.Embedded = 2
				break
			}
		}
		ms = append(ms, f)
	}

	for _, f := range t.Methods().Slice() {
		f.Sym.SetUniq(false)
	}

	ms = append(ms, t.Methods().Slice()...)
	sort.Sort(methcmp(ms))
	t.AllMethods().Set(ms)
}

// Given funarg struct list, return list of ODCLFIELD Node fn args.
func structargs(tl *types.Type, mustname bool) []ir.Node {
	var args []ir.Node
	gen := 0
	for _, t := range tl.Fields().Slice() {
		s := t.Sym
		if mustname && (s == nil || s.Name == "_") {
			// invent a name so that we can refer to it in the trampoline
			s = lookupN(".anon", gen)
			gen++
		}
		a := symfield(s, t.Type)
		a.SetPos(t.Pos)
		a.SetIsDDD(t.IsDDD())
		args = append(args, a)
	}

	return args
}

// Generate a wrapper function to convert from
// a receiver of type T to a receiver of type U.
// That is,
//
//	func (t T) M() {
//		...
//	}
//
// already exists; this function generates
//
//	func (u U) M() {
//		u.M()
//	}
//
// where the types T and U are such that u.M() is valid
// and calls the T.M method.
// The resulting function is for use in method tables.
//
//	rcvr - U
//	method - M func (t T)(), a TFIELD type struct
//	newnam - the eventual mangled name of this function
func genwrapper(rcvr *types.Type, method *types.Field, newnam *types.Sym) {
	if false && base.Flag.LowerR != 0 {
		fmt.Printf("genwrapper rcvrtype=%v method=%v newnam=%v\n", rcvr, method, newnam)
	}

	// Only generate (*T).M wrappers for T.M in T's own package.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type &&
		rcvr.Elem().Sym != nil && rcvr.Elem().Sym.Pkg != ir.LocalPkg {
		return
	}

	// Only generate I.M wrappers for I in I's own package
	// but keep doing it for error.Error (was issue #29304).
	if rcvr.IsInterface() && rcvr.Sym != nil && rcvr.Sym.Pkg != ir.LocalPkg && rcvr != types.Errortype {
		return
	}

	base.Pos = autogeneratedPos
	dclcontext = ir.PEXTERN

	tfn := ir.Nod(ir.OTFUNC, nil, nil)
	tfn.SetLeft(namedfield(".this", rcvr))
	tfn.PtrList().Set(structargs(method.Type.Params(), true))
	tfn.PtrRlist().Set(structargs(method.Type.Results(), false))

	fn := dclfunc(newnam, tfn)
	fn.SetDupok(true)

	nthis := ir.AsNode(tfn.Type().Recv().Nname)

	methodrcvr := method.Type.Recv().Type

	// generate nil pointer check for better error
	if rcvr.IsPtr() && rcvr.Elem() == methodrcvr {
		// generating wrapper from *T to T.
		n := ir.Nod(ir.OIF, nil, nil)
		n.SetLeft(ir.Nod(ir.OEQ, nthis, nodnil()))
		call := ir.Nod(ir.OCALL, syslook("panicwrap"), nil)
		n.PtrBody().Set1(call)
		fn.PtrBody().Append(n)
	}

	dot := adddot(nodSym(ir.OXDOT, nthis, method.Sym))

	// generate call
	// It's not possible to use a tail call when dynamic linking on ppc64le. The
	// bad scenario is when a local call is made to the wrapper: the wrapper will
	// call the implementation, which might be in a different module and so set
	// the TOC to the appropriate value for that module. But if it returns
	// directly to the wrapper's caller, nothing will reset it to the correct
	// value for that function.
	if !instrumenting && rcvr.IsPtr() && methodrcvr.IsPtr() && method.Embedded != 0 && !isifacemethod(method.Type) && !(thearch.LinkArch.Name == "ppc64le" && base.Ctxt.Flag_dynlink) {
		// generate tail call: adjust pointer receiver and jump to embedded method.
		dot = dot.Left() // skip final .M
		// TODO(mdempsky): Remove dependency on dotlist.
		if !dotlist[0].field.Type.IsPtr() {
			dot = ir.Nod(ir.OADDR, dot, nil)
		}
		as := ir.Nod(ir.OAS, nthis, convnop(dot, rcvr))
		fn.PtrBody().Append(as)
		fn.PtrBody().Append(nodSym(ir.ORETJMP, nil, methodSym(methodrcvr, method.Sym)))
	} else {
		fn.SetWrapper(true) // ignore frame for panic+recover matching
		call := ir.Nod(ir.OCALL, dot, nil)
		call.PtrList().Set(paramNnames(tfn.Type()))
		call.SetIsDDD(tfn.Type().IsVariadic())
		if method.Type.NumResults() > 0 {
			n := ir.Nod(ir.ORETURN, nil, nil)
			n.PtrList().Set1(call)
			call = n
		}
		fn.PtrBody().Append(call)
	}

	if false && base.Flag.LowerR != 0 {
		ir.DumpList("genwrapper body", fn.Body())
	}

	funcbody()
	if base.Debug.DclStack != 0 {
		testdclstack()
	}

	typecheckFunc(fn)
	Curfn = fn
	typecheckslice(fn.Body().Slice(), ctxStmt)

	// Inline calls within (*T).M wrappers. This is safe because we only
	// generate those wrappers within the same compilation unit as (T).M.
	// TODO(mdempsky): Investigate why we can't enable this more generally.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type && rcvr.Elem().Sym != nil {
		inlcalls(fn)
	}
	escapeFuncs([]*ir.Func{fn}, false)

	Curfn = nil
	xtop = append(xtop, fn)
}

func paramNnames(ft *types.Type) []ir.Node {
	args := make([]ir.Node, ft.NumParams())
	for i, f := range ft.Params().FieldSlice() {
		args[i] = ir.AsNode(f.Nname)
	}
	return args
}

func hashmem(t *types.Type) ir.Node {
	sym := Runtimepkg.Lookup("memhash")

	n := NewName(sym)
	setNodeNameFunc(n)
	n.SetType(functype(nil, []ir.Node{
		anonfield(types.NewPtr(t)),
		anonfield(types.Types[types.TUINTPTR]),
		anonfield(types.Types[types.TUINTPTR]),
	}, []ir.Node{
		anonfield(types.Types[types.TUINTPTR]),
	}))
	return n
}

func ifacelookdot(s *types.Sym, t *types.Type, ignorecase bool) (m *types.Field, followptr bool) {
	if t == nil {
		return nil, false
	}

	path, ambig := dotpath(s, t, &m, ignorecase)
	if path == nil {
		if ambig {
			base.Errorf("%v.%v is ambiguous", t, s)
		}
		return nil, false
	}

	for _, d := range path {
		if d.field.Type.IsPtr() {
			followptr = true
			break
		}
	}

	if !m.IsMethod() {
		base.Errorf("%v.%v is a field, not a method", t, s)
		return nil, followptr
	}

	return m, followptr
}

func implements(t, iface *types.Type, m, samename **types.Field, ptr *int) bool {
	t0 := t
	if t == nil {
		return false
	}

	if t.IsInterface() {
		i := 0
		tms := t.Fields().Slice()
		for _, im := range iface.Fields().Slice() {
			for i < len(tms) && tms[i].Sym != im.Sym {
				i++
			}
			if i == len(tms) {
				*m = im
				*samename = nil
				*ptr = 0
				return false
			}
			tm := tms[i]
			if !types.Identical(tm.Type, im.Type) {
				*m = im
				*samename = tm
				*ptr = 0
				return false
			}
		}

		return true
	}

	t = methtype(t)
	var tms []*types.Field
	if t != nil {
		expandmeth(t)
		tms = t.AllMethods().Slice()
	}
	i := 0
	for _, im := range iface.Fields().Slice() {
		if im.Broke() {
			continue
		}
		for i < len(tms) && tms[i].Sym != im.Sym {
			i++
		}
		if i == len(tms) {
			*m = im
			*samename, _ = ifacelookdot(im.Sym, t, true)
			*ptr = 0
			return false
		}
		tm := tms[i]
		if tm.Nointerface() || !types.Identical(tm.Type, im.Type) {
			*m = im
			*samename = tm
			*ptr = 0
			return false
		}
		followptr := tm.Embedded == 2

		// if pointer receiver in method,
		// the method does not exist for value types.
		rcvr := tm.Type.Recv().Type
		if rcvr.IsPtr() && !t0.IsPtr() && !followptr && !isifacemethod(tm.Type) {
			if false && base.Flag.LowerR != 0 {
				base.Errorf("interface pointer mismatch")
			}

			*m = im
			*samename = nil
			*ptr = 1
			return false
		}
	}

	// We're going to emit an OCONVIFACE.
	// Call itabname so that (t, iface)
	// gets added to itabs early, which allows
	// us to de-virtualize calls through this
	// type/interface pair later. See peekitabs in reflect.go
	if isdirectiface(t0) && !iface.IsEmptyInterface() {
		itabname(t0, iface)
	}
	return true
}

func liststmt(l []ir.Node) ir.Node {
	n := ir.Nod(ir.OBLOCK, nil, nil)
	n.PtrList().Set(l)
	if len(l) != 0 {
		n.SetPos(l[0].Pos())
	}
	return n
}

func ngotype(n ir.Node) *types.Sym {
	if n.Type() != nil {
		return typenamesym(n.Type())
	}
	return nil
}

// The result of addinit MUST be assigned back to n, e.g.
// 	n.Left = addinit(n.Left, init)
func addinit(n ir.Node, init []ir.Node) ir.Node {
	if len(init) == 0 {
		return n
	}
	if ir.MayBeShared(n) {
		// Introduce OCONVNOP to hold init list.
		n = ir.Nod(ir.OCONVNOP, n, nil)
		n.SetType(n.Left().Type())
		n.SetTypecheck(1)
	}

	n.PtrInit().Prepend(init...)
	n.SetHasCall(true)
	return n
}

// The linker uses the magic symbol prefixes "go." and "type."
// Avoid potential confusion between import paths and symbols
// by rejecting these reserved imports for now. Also, people
// "can do weird things in GOPATH and we'd prefer they didn't
// do _that_ weird thing" (per rsc). See also #4257.
var reservedimports = []string{
	"go",
	"type",
}

func isbadimport(path string, allowSpace bool) bool {
	if strings.Contains(path, "\x00") {
		base.Errorf("import path contains NUL")
		return true
	}

	for _, ri := range reservedimports {
		if path == ri {
			base.Errorf("import path %q is reserved and cannot be used", path)
			return true
		}
	}

	for _, r := range path {
		if r == utf8.RuneError {
			base.Errorf("import path contains invalid UTF-8 sequence: %q", path)
			return true
		}

		if r < 0x20 || r == 0x7f {
			base.Errorf("import path contains control character: %q", path)
			return true
		}

		if r == '\\' {
			base.Errorf("import path contains backslash; use slash: %q", path)
			return true
		}

		if !allowSpace && unicode.IsSpace(r) {
			base.Errorf("import path contains space character: %q", path)
			return true
		}

		if strings.ContainsRune("!\"#$%&'()*,:;<=>?[]^`{|}", r) {
			base.Errorf("import path contains invalid character '%c': %q", r, path)
			return true
		}
	}

	return false
}

// Can this type be stored directly in an interface word?
// Yes, if the representation is a single pointer.
func isdirectiface(t *types.Type) bool {
	if t.Broke() {
		return false
	}

	switch t.Etype {
	case types.TPTR:
		// Pointers to notinheap types must be stored indirectly. See issue 42076.
		return !t.Elem().NotInHeap()
	case types.TCHAN,
		types.TMAP,
		types.TFUNC,
		types.TUNSAFEPTR:
		return true

	case types.TARRAY:
		// Array of 1 direct iface type can be direct.
		return t.NumElem() == 1 && isdirectiface(t.Elem())

	case types.TSTRUCT:
		// Struct with 1 field of direct iface type can be direct.
		return t.NumFields() == 1 && isdirectiface(t.Field(0).Type)
	}

	return false
}

// itabType loads the _type field from a runtime.itab struct.
func itabType(itab ir.Node) ir.Node {
	typ := nodSym(ir.ODOTPTR, itab, nil)
	typ.SetType(types.NewPtr(types.Types[types.TUINT8]))
	typ.SetTypecheck(1)
	typ.SetOffset(int64(Widthptr)) // offset of _type in runtime.itab
	typ.SetBounded(true)           // guaranteed not to fault
	return typ
}

// ifaceData loads the data field from an interface.
// The concrete type must be known to have type t.
// It follows the pointer if !isdirectiface(t).
func ifaceData(pos src.XPos, n ir.Node, t *types.Type) ir.Node {
	if t.IsInterface() {
		base.Fatalf("ifaceData interface: %v", t)
	}
	ptr := nodlSym(pos, ir.OIDATA, n, nil)
	if isdirectiface(t) {
		ptr.SetType(t)
		ptr.SetTypecheck(1)
		return ptr
	}
	ptr.SetType(types.NewPtr(t))
	ptr.SetTypecheck(1)
	ind := ir.NodAt(pos, ir.ODEREF, ptr, nil)
	ind.SetType(t)
	ind.SetTypecheck(1)
	ind.SetBounded(true)
	return ind
}

// typePos returns the position associated with t.
// This is where t was declared or where it appeared as a type expression.
func typePos(t *types.Type) src.XPos {
	n := ir.AsNode(t.Nod)
	if n == nil || !n.Pos().IsKnown() {
		base.Fatalf("bad type: %v", t)
	}
	return n.Pos()
}
