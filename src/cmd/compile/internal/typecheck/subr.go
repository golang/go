// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"bytes"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

func AssignConv(n ir.Node, t *types.Type, context string) ir.Node {
	return assignconvfn(n, t, func() string { return context })
}

// DotImportRefs maps idents introduced by importDot back to the
// ir.PkgName they were dot-imported through.
var DotImportRefs map[*ir.Ident]*ir.PkgName

// LookupNum looks up the symbol starting with prefix and ending with
// the decimal n. If prefix is too long, LookupNum panics.
func LookupNum(prefix string, n int) *types.Sym {
	var buf [20]byte // plenty long enough for all current users
	copy(buf[:], prefix)
	b := strconv.AppendInt(buf[:len(prefix)], int64(n), 10)
	return types.LocalPkg.LookupBytes(b)
}

// Given funarg struct list, return list of fn args.
func NewFuncParams(tl *types.Type, mustname bool) []*ir.Field {
	var args []*ir.Field
	gen := 0
	for _, t := range tl.Fields().Slice() {
		s := t.Sym
		if mustname && (s == nil || s.Name == "_") {
			// invent a name so that we can refer to it in the trampoline
			s = LookupNum(".anon", gen)
			gen++
		} else if s != nil && s.Pkg != types.LocalPkg {
			// TODO(mdempsky): Preserve original position, name, and package.
			s = Lookup(s.Name)
		}
		a := ir.NewField(base.Pos, s, nil, t.Type)
		a.Pos = t.Pos
		a.IsDDD = t.IsDDD()
		args = append(args, a)
	}

	return args
}

// newname returns a new ONAME Node associated with symbol s.
func NewName(s *types.Sym) *ir.Name {
	n := ir.NewNameAt(base.Pos, s)
	n.Curfn = ir.CurFunc
	return n
}

// NodAddr returns a node representing &n at base.Pos.
func NodAddr(n ir.Node) *ir.AddrExpr {
	return NodAddrAt(base.Pos, n)
}

// nodAddrPos returns a node representing &n at position pos.
func NodAddrAt(pos src.XPos, n ir.Node) *ir.AddrExpr {
	n = markAddrOf(n)
	return ir.NewAddrExpr(pos, n)
}

func markAddrOf(n ir.Node) ir.Node {
	if IncrementalAddrtaken {
		// We can only do incremental addrtaken computation when it is ok
		// to typecheck the argument of the OADDR. That's only safe after the
		// main typecheck has completed, and not loading the inlined body.
		// The argument to OADDR needs to be typechecked because &x[i] takes
		// the address of x if x is an array, but not if x is a slice.
		// Note: OuterValue doesn't work correctly until n is typechecked.
		n = typecheck(n, ctxExpr)
		if x := ir.OuterValue(n); x.Op() == ir.ONAME {
			x.Name().SetAddrtaken(true)
		}
	} else {
		// Remember that we built an OADDR without computing the Addrtaken bit for
		// its argument. We'll do that later in bulk using computeAddrtaken.
		DirtyAddrtaken = true
	}
	return n
}

// If IncrementalAddrtaken is false, we do not compute Addrtaken for an OADDR Node
// when it is built. The Addrtaken bits are set in bulk by computeAddrtaken.
// If IncrementalAddrtaken is true, then when an OADDR Node is built the Addrtaken
// field of its argument is updated immediately.
var IncrementalAddrtaken = false

// If DirtyAddrtaken is true, then there are OADDR whose corresponding arguments
// have not yet been marked as Addrtaken.
var DirtyAddrtaken = false

func ComputeAddrtaken(top []ir.Node) {
	for _, n := range top {
		var doVisit func(n ir.Node)
		doVisit = func(n ir.Node) {
			if n.Op() == ir.OADDR {
				if x := ir.OuterValue(n.(*ir.AddrExpr).X); x.Op() == ir.ONAME {
					x.Name().SetAddrtaken(true)
					if x.Name().IsClosureVar() {
						// Mark the original variable as Addrtaken so that capturevars
						// knows not to pass it by value.
						x.Name().Defn.Name().SetAddrtaken(true)
					}
				}
			}
			if n.Op() == ir.OCLOSURE {
				ir.VisitList(n.(*ir.ClosureExpr).Func.Body, doVisit)
			}
		}
		ir.Visit(n, doVisit)
	}
}

func NodNil() ir.Node {
	n := ir.NewNilExpr(base.Pos)
	n.SetType(types.Types[types.TNIL])
	return n
}

// AddImplicitDots finds missing fields in obj.field that
// will give the shortest unique addressing and
// modifies the tree with missing field names.
func AddImplicitDots(n *ir.SelectorExpr) *ir.SelectorExpr {
	n.X = typecheck(n.X, ctxType|ctxExpr)
	if n.X.Diag() {
		n.SetDiag(true)
	}
	t := n.X.Type()
	if t == nil {
		return n
	}

	if n.X.Op() == ir.OTYPE {
		return n
	}

	s := n.Sel
	if s == nil {
		return n
	}

	switch path, ambig := dotpath(s, t, nil, false); {
	case path != nil:
		// rebuild elided dots
		for c := len(path) - 1; c >= 0; c-- {
			dot := ir.NewSelectorExpr(n.Pos(), ir.ODOT, n.X, path[c].field.Sym)
			dot.SetImplicit(true)
			dot.SetType(path[c].field.Type)
			n.X = dot
		}
	case ambig:
		base.Errorf("ambiguous selector %v", n)
		n.X = nil
	}

	return n
}

// CalcMethods calculates all the methods (including embedding) of a non-interface
// type t.
func CalcMethods(t *types.Type) {
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
	sort.Sort(types.MethodsByName(ms))
	t.SetAllMethods(ms)
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

	var fields *types.Fields
	if u.IsStruct() {
		fields = u.Fields()
	} else {
		fields = u.AllMethods()
	}
	for _, f := range fields.Slice() {
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

// dotlist is used by adddot1 to record the path of embedded fields
// used to access a target field or method.
// Must be non-nil so that dotpath returns a non-nil slice even if d is zero.
var dotlist = make([]dlist, 10)

// Convert node n for assignment to type t.
func assignconvfn(n ir.Node, t *types.Type, context func() string) ir.Node {
	if n == nil || n.Type() == nil || n.Type().Broke() {
		return n
	}

	if t.Kind() == types.TBLANK && n.Type().Kind() == types.TNIL {
		base.Errorf("use of untyped nil")
	}

	n = convlit1(n, t, false, context)
	if n.Type() == nil {
		return n
	}
	if t.Kind() == types.TBLANK {
		return n
	}

	// Convert ideal bool from comparison to plain bool
	// if the next step is non-bool (like interface{}).
	if n.Type() == types.UntypedBool && !t.IsBoolean() {
		if n.Op() == ir.ONAME || n.Op() == ir.OLITERAL {
			r := ir.NewConvExpr(base.Pos, ir.OCONVNOP, nil, n)
			r.SetType(types.Types[types.TBOOL])
			r.SetTypecheck(1)
			r.SetImplicit(true)
			n = r
		}
	}

	if types.Identical(n.Type(), t) {
		return n
	}

	op, why := Assignop(n.Type(), t)
	if op == ir.OXXX {
		base.Errorf("cannot use %L as type %v in %s%s", n, t, context(), why)
		op = ir.OCONV
	}

	r := ir.NewConvExpr(base.Pos, op, t, n)
	r.SetTypecheck(1)
	r.SetImplicit(true)
	return r
}

// Is type src assignment compatible to type dst?
// If so, return op code to use in conversion.
// If not, return OXXX. In this case, the string return parameter may
// hold a reason why. In all other cases, it'll be the empty string.
func Assignop(src, dst *types.Type) (ir.Op, string) {
	if src == dst {
		return ir.OCONVNOP, ""
	}
	if src == nil || dst == nil || src.Kind() == types.TFORW || dst.Kind() == types.TFORW || src.Underlying() == nil || dst.Underlying() == nil {
		return ir.OXXX, ""
	}

	// 1. src type is identical to dst.
	if types.Identical(src, dst) {
		return ir.OCONVNOP, ""
	}
	return Assignop1(src, dst)
}

func Assignop1(src, dst *types.Type) (ir.Op, string) {
	// 2. src and dst have identical underlying types and
	//   a. either src or dst is not a named type, or
	//   b. both are empty interface types, or
	//   c. at least one is a gcshape type.
	// For assignable but different non-empty interface types,
	// we want to recompute the itab. Recomputing the itab ensures
	// that itabs are unique (thus an interface with a compile-time
	// type I has an itab with interface type I).
	if types.Identical(src.Underlying(), dst.Underlying()) {
		if src.IsEmptyInterface() {
			// Conversion between two empty interfaces
			// requires no code.
			return ir.OCONVNOP, ""
		}
		if (src.Sym() == nil || dst.Sym() == nil) && !src.IsInterface() {
			// Conversion between two types, at least one unnamed,
			// needs no conversion. The exception is nonempty interfaces
			// which need to have their itab updated.
			return ir.OCONVNOP, ""
		}
		if src.IsShape() || dst.IsShape() {
			// Conversion between a shape type and one of the types
			// it represents also needs no conversion.
			return ir.OCONVNOP, ""
		}
	}

	// 3. dst is an interface type and src implements dst.
	if dst.IsInterface() && src.Kind() != types.TNIL {
		var missing, have *types.Field
		var ptr int
		if src.IsShape() {
			// Shape types implement things they have already
			// been typechecked to implement, even if they
			// don't have the methods for them.
			return ir.OCONVIFACE, ""
		}
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
				"\t\thave %v%S\n\t\twant %v%S", src, dst, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
		} else if ptr != 0 {
			why = fmt.Sprintf(":\n\t%v does not implement %v (%v method has pointer receiver)", src, dst, missing.Sym)
		} else if have != nil {
			why = fmt.Sprintf(":\n\t%v does not implement %v (missing %v method)\n"+
				"\t\thave %v%S\n\t\twant %v%S", src, dst, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
		} else {
			why = fmt.Sprintf(":\n\t%v does not implement %v (missing %v method)", src, dst, missing.Sym)
		}

		return ir.OXXX, why
	}

	if isptrto(dst, types.TINTER) {
		why := fmt.Sprintf(":\n\t%v is pointer to interface, not interface", dst)
		return ir.OXXX, why
	}

	if src.IsInterface() && dst.Kind() != types.TBLANK {
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
		if types.Identical(src.Elem(), dst.Elem()) && (src.Sym() == nil || dst.Sym() == nil) {
			return ir.OCONVNOP, ""
		}
	}

	// 5. src is the predeclared identifier nil and dst is a nillable type.
	if src.Kind() == types.TNIL {
		switch dst.Kind() {
		case types.TPTR,
			types.TFUNC,
			types.TMAP,
			types.TCHAN,
			types.TINTER,
			types.TSLICE:
			return ir.OCONVNOP, ""
		}
	}

	// 6. rule about untyped constants - already converted by DefaultLit.

	// 7. Any typed value can be assigned to the blank identifier.
	if dst.Kind() == types.TBLANK {
		return ir.OCONVNOP, ""
	}

	return ir.OXXX, ""
}

// Can we convert a value of type src to a value of type dst?
// If so, return op code to use in conversion (maybe OCONVNOP).
// If not, return OXXX. In this case, the string return parameter may
// hold a reason why. In all other cases, it'll be the empty string.
// srcConstant indicates whether the value of type src is a constant.
func Convertop(srcConstant bool, src, dst *types.Type) (ir.Op, string) {
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
	if src.IsString() && dst.IsSlice() && dst.Elem().NotInHeap() && (dst.Elem().Kind() == types.ByteType.Kind() || dst.Elem().Kind() == types.RuneType.Kind()) {
		why := fmt.Sprintf(":\n\t%v is incomplete (or unallocatable)", dst.Elem())
		return ir.OXXX, why
	}

	// 1. src can be assigned to dst.
	op, why := Assignop(src, dst)
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
	if types.IdenticalIgnoreTags(src.Underlying(), dst.Underlying()) {
		return ir.OCONVNOP, ""
	}

	// 3. src and dst are unnamed pointer types and, ignoring struct tags,
	// their base types have identical underlying types.
	if src.IsPtr() && dst.IsPtr() && src.Sym() == nil && dst.Sym() == nil {
		if types.IdenticalIgnoreTags(src.Elem().Underlying(), dst.Elem().Underlying()) {
			return ir.OCONVNOP, ""
		}
	}

	// 4. src and dst are both integer or floating point types.
	if (src.IsInteger() || src.IsFloat()) && (dst.IsInteger() || dst.IsFloat()) {
		if types.SimType[src.Kind()] == types.SimType[dst.Kind()] {
			return ir.OCONVNOP, ""
		}
		return ir.OCONV, ""
	}

	// 5. src and dst are both complex types.
	if src.IsComplex() && dst.IsComplex() {
		if types.SimType[src.Kind()] == types.SimType[dst.Kind()] {
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
		if src.Elem().Kind() == types.ByteType.Kind() {
			return ir.OBYTES2STR, ""
		}
		if src.Elem().Kind() == types.RuneType.Kind() {
			return ir.ORUNES2STR, ""
		}
	}

	// 7. src is a string and dst is []byte or []rune.
	// String to slice.
	if src.IsString() && dst.IsSlice() {
		if dst.Elem().Kind() == types.ByteType.Kind() {
			return ir.OSTR2BYTES, ""
		}
		if dst.Elem().Kind() == types.RuneType.Kind() {
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

	// 10. src is map and dst is a pointer to corresponding hmap.
	// This rule is needed for the implementation detail that
	// go gc maps are implemented as a pointer to a hmap struct.
	if src.Kind() == types.TMAP && dst.IsPtr() &&
		src.MapType().Hmap == dst.Elem() {
		return ir.OCONVNOP, ""
	}

	// 11. src is a slice and dst is a pointer-to-array.
	// They must have same element type.
	if src.IsSlice() && dst.IsPtr() && dst.Elem().IsArray() &&
		types.Identical(src.Elem(), dst.Elem().Elem()) {
		if !types.AllowsGoVersion(curpkg(), 1, 17) {
			return ir.OXXX, ":\n\tconversion of slices to array pointers only supported as of -lang=go1.17"
		}
		return ir.OSLICE2ARRPTR, ""
	}

	return ir.OXXX, ""
}

// Code to resolve elided DOTs in embedded types.

// A dlist stores a pointer to a TFIELD Type embedded within
// a TSTRUCT or TINTER Type.
type dlist struct {
	field *types.Field
}

// dotpath computes the unique shortest explicit selector path to fully qualify
// a selection expression x.f, where x is of type t and f is the symbol s.
// If no such path exists, dotpath returns nil.
// If there are multiple shortest paths to the same depth, ambig is true.
func dotpath(s *types.Sym, t *types.Type, save **types.Field, ignorecase bool) (path []dlist, ambig bool) {
	// The embedding of types within structs imposes a tree structure onto
	// types: structs parent the types they embed, and types parent their
	// fields or methods. Our goal here is to find the shortest path to
	// a field or method named s in the subtree rooted at t. To accomplish
	// that, we iteratively perform depth-first searches of increasing depth
	// until we either find the named field/method or exhaust the tree.
	for d := 0; ; d++ {
		if d > len(dotlist) {
			dotlist = append(dotlist, dlist{})
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

func expand0(t *types.Type) {
	u := t
	if u.IsPtr() {
		u = u.Elem()
	}

	if u.IsInterface() {
		for _, f := range u.AllMethods().Slice() {
			if f.Sym.Uniq() {
				continue
			}
			f.Sym.SetUniq(true)
			slist = append(slist, symlink{field: f})
		}

		return
	}

	u = types.ReceiverBaseType(t)
	if u != nil {
		for _, f := range u.Methods().Slice() {
			if f.Sym.Uniq() {
				continue
			}
			f.Sym.SetUniq(true)
			slist = append(slist, symlink{field: f})
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
		var fields *types.Fields
		if u.IsStruct() {
			fields = u.Fields()
		} else {
			fields = u.AllMethods()
		}
		for _, f := range fields.Slice() {
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

// implements reports whether t implements the interface iface. t can be
// an interface, a type parameter, or a concrete type. If implements returns
// false, it stores a method of iface that is not implemented in *m. If the
// method name matches but the type is wrong, it additionally stores the type
// of the method (on t) in *samename.
func implements(t, iface *types.Type, m, samename **types.Field, ptr *int) bool {
	t0 := t
	if t == nil {
		return false
	}

	if t.IsInterface() || t.IsTypeParam() {
		if t.IsTypeParam() {
			// If t is a simple type parameter T, its type and underlying is the same.
			// If t is a type definition:'type P[T any] T', its type is P[T] and its
			// underlying is T. Therefore we use 't.Underlying() != t' to distinguish them.
			if t.Underlying() != t {
				CalcMethods(t)
			} else {
				// A typeparam satisfies an interface if its type bound
				// has all the methods of that interface.
				t = t.Bound()
			}
		}
		i := 0
		tms := t.AllMethods().Slice()
		for _, im := range iface.AllMethods().Slice() {
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

	t = types.ReceiverBaseType(t)
	var tms []*types.Field
	if t != nil {
		CalcMethods(t)
		tms = t.AllMethods().Slice()
	}
	i := 0
	for _, im := range iface.AllMethods().Slice() {
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
		if rcvr.IsPtr() && !t0.IsPtr() && !followptr && !types.IsInterfaceMethod(tm.Type) {
			if false && base.Flag.LowerR != 0 {
				base.Errorf("interface pointer mismatch")
			}

			*m = im
			*samename = nil
			*ptr = 1
			return false
		}
	}

	return true
}

func isptrto(t *types.Type, et types.Kind) bool {
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
	if t.Kind() != et {
		return false
	}
	return true
}

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
		var fields *types.Fields
		if u.IsStruct() {
			fields = u.Fields()
		} else {
			fields = u.AllMethods()
		}
		for _, f := range fields.Slice() {
			if f.Sym == s || (ignorecase && f.IsMethod() && strings.EqualFold(f.Sym.Name, s.Name)) {
				if save != nil {
					*save = f
				}
				c++
			}
		}
	}

	u = t
	if t.Sym() != nil && t.IsPtr() && !t.Elem().IsPtr() {
		// If t is a defined pointer type, then x.m is shorthand for (*x).m.
		u = t.Elem()
	}
	u = types.ReceiverBaseType(u)
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

var slist []symlink

// Code to help generate trampoline functions for methods on embedded
// types. These are approx the same as the corresponding AddImplicitDots
// routines except that they expect to be called with unique tasks and
// they return the actual methods.

type symlink struct {
	field *types.Field
}

// TypesOf converts a list of nodes to a list
// of types of those nodes.
func TypesOf(x []ir.Node) []*types.Type {
	r := make([]*types.Type, len(x))
	for i, n := range x {
		r[i] = n.Type()
	}
	return r
}

// addTargs writes out the targs to buffer b as a comma-separated list enclosed by
// brackets.
func addTargs(b *bytes.Buffer, targs []*types.Type) {
	b.WriteByte('[')
	for i, targ := range targs {
		if i > 0 {
			b.WriteByte(',')
		}
		// Make sure that type arguments (including type params), are
		// uniquely specified. LinkString() eliminates all spaces
		// and includes the package path (local package path is "" before
		// linker substitution).
		tstring := targ.LinkString()
		b.WriteString(tstring)
	}
	b.WriteString("]")
}

// InstTypeName creates a name for an instantiated type, based on the name of the
// generic type and the type args.
func InstTypeName(name string, targs []*types.Type) string {
	b := bytes.NewBufferString(name)
	addTargs(b, targs)
	return b.String()
}

// makeInstName1 returns the name of the generic function instantiated with the
// given types, which can have type params or shapes, or be concrete types. name is
// the name of the generic function or method.
func makeInstName1(name string, targs []*types.Type, hasBrackets bool) string {
	b := bytes.NewBufferString("")
	i := strings.Index(name, "[")
	assert(hasBrackets == (i >= 0))
	if i >= 0 {
		b.WriteString(name[0:i])
	} else {
		b.WriteString(name)
	}
	addTargs(b, targs)
	if i >= 0 {
		i2 := strings.LastIndex(name[i:], "]")
		assert(i2 >= 0)
		b.WriteString(name[i+i2+1:])
	}
	return b.String()
}

// MakeFuncInstSym makes the unique sym for a stenciled generic function or method,
// based on the name of the function gf and the targs. It replaces any
// existing bracket type list in the name. MakeInstName asserts that gf has
// brackets in its name if and only if hasBrackets is true.
//
// Names of declared generic functions have no brackets originally, so hasBrackets
// should be false. Names of generic methods already have brackets, since the new
// type parameter is specified in the generic type of the receiver (e.g. func
// (func (v *value[T]).set(...) { ... } has the original name (*value[T]).set.
//
// The standard naming is something like: 'genFn[int,bool]' for functions and
// '(*genType[int,bool]).methodName' for methods
//
// isMethodNode specifies if the name of a method node is being generated (as opposed
// to a name of an instantiation of generic function or name of the shape-based
// function that helps implement a method of an instantiated type). For method nodes
// on shape types, we prepend "nofunc.", because method nodes for shape types will
// have no body, and we want to avoid a name conflict with the shape-based function
// that helps implement the same method for fully-instantiated types. Function names
// are also created at the end of (*Tsubster).typ1, so we append "nofunc" there as
// well, as needed.
func MakeFuncInstSym(gf *types.Sym, targs []*types.Type, isMethodNode, hasBrackets bool) *types.Sym {
	nm := makeInstName1(gf.Name, targs, hasBrackets)
	if targs[0].HasShape() && isMethodNode {
		nm = "nofunc." + nm
	}
	return gf.Pkg.Lookup(nm)
}

func MakeDictSym(gf *types.Sym, targs []*types.Type, hasBrackets bool) *types.Sym {
	for _, targ := range targs {
		if targ.HasTParam() {
			fmt.Printf("FUNCTION %s\n", gf.Name)
			for _, targ := range targs {
				fmt.Printf("  PARAM %+v\n", targ)
			}
			panic("dictionary should always have concrete type args")
		}
	}
	name := makeInstName1(gf.Name, targs, hasBrackets)
	name = fmt.Sprintf("%s.%s", objabi.GlobalDictPrefix, name)
	return gf.Pkg.Lookup(name)
}

func assert(p bool) {
	base.Assert(p)
}

// List of newly fully-instantiated types who should have their methods generated.
var instTypeList []*types.Type

// NeedInstType adds a new fully-instantiated type to instTypeList.
func NeedInstType(t *types.Type) {
	instTypeList = append(instTypeList, t)
}

// GetInstTypeList returns the current contents of instTypeList.
func GetInstTypeList() []*types.Type {
	r := instTypeList
	return r
}

// ClearInstTypeList clears the contents of instTypeList.
func ClearInstTypeList() {
	instTypeList = nil
}

// General type substituter, for replacing typeparams with type args.
type Tsubster struct {
	Tparams []*types.Type
	Targs   []*types.Type
	// If non-nil, the substitution map from name nodes in the generic function to the
	// name nodes in the new stenciled function.
	Vars map[*ir.Name]*ir.Name
	// If non-nil, function to substitute an incomplete (TFORW) type.
	SubstForwFunc func(*types.Type) *types.Type
}

// Typ computes the type obtained by substituting any type parameter or shape in t
// that appears in subst.Tparams with the corresponding type argument in subst.Targs.
// If t contains no type parameters, the result is t; otherwise the result is a new
// type. It deals with recursive types by using TFORW types and finding partially or
// fully created types via sym.Def.
func (ts *Tsubster) Typ(t *types.Type) *types.Type {
	// Defer the CheckSize calls until we have fully-defined
	// (possibly-recursive) top-level type.
	types.DeferCheckSize()
	r := ts.typ1(t)
	types.ResumeCheckSize()
	return r
}

func (ts *Tsubster) typ1(t *types.Type) *types.Type {
	if !t.HasTParam() && !t.HasShape() && t.Kind() != types.TFUNC {
		// Note: function types need to be copied regardless, as the
		// types of closures may contain declarations that need
		// to be copied. See #45738.
		return t
	}

	if t.IsTypeParam() || t.IsShape() {
		for i, tp := range ts.Tparams {
			if tp == t {
				return ts.Targs[i]
			}
		}
		// If t is a simple typeparam T, then t has the name/symbol 'T'
		// and t.Underlying() == t.
		//
		// However, consider the type definition: 'type P[T any] T'. We
		// might use this definition so we can have a variant of type T
		// that we can add new methods to. Suppose t is a reference to
		// P[T]. t has the name 'P[T]', but its kind is TTYPEPARAM,
		// because P[T] is defined as T. If we look at t.Underlying(), it
		// is different, because the name of t.Underlying() is 'T' rather
		// than 'P[T]'. But the kind of t.Underlying() is also TTYPEPARAM.
		// In this case, we do the needed recursive substitution in the
		// case statement below.
		if t.Underlying() == t {
			// t is a simple typeparam that didn't match anything in tparam
			return t
		}
		// t is a more complex typeparam (e.g. P[T], as above, whose
		// definition is just T).
		assert(t.Sym() != nil)
	}

	var newsym *types.Sym
	var neededTargs []*types.Type
	var targsChanged bool
	var forw *types.Type

	if t.Sym() != nil && (t.HasTParam() || t.HasShape()) {
		// Need to test for t.HasTParam() again because of special TFUNC case above.
		// Translate the type params for this type according to
		// the tparam/targs mapping from subst.
		neededTargs = make([]*types.Type, len(t.RParams()))
		for i, rparam := range t.RParams() {
			neededTargs[i] = ts.typ1(rparam)
			if !types.IdenticalStrict(neededTargs[i], rparam) {
				targsChanged = true
			}
		}
		// For a named (defined) type, we have to change the name of the
		// type as well. We do this first, so we can look up if we've
		// already seen this type during this substitution or other
		// definitions/substitutions.
		genName := genericTypeName(t.Sym())
		newsym = t.Sym().Pkg.Lookup(InstTypeName(genName, neededTargs))
		if newsym.Def != nil {
			// We've already created this instantiated defined type.
			return newsym.Def.Type()
		}

		// In order to deal with recursive generic types, create a TFORW
		// type initially and set the Def field of its sym, so it can be
		// found if this type appears recursively within the type.
		forw = NewIncompleteNamedType(t.Pos(), newsym)
		//println("Creating new type by sub", newsym.Name, forw.HasTParam())
		forw.SetRParams(neededTargs)
		// Copy the OrigSym from the re-instantiated type (which is the sym of
		// the base generic type).
		assert(t.OrigSym() != nil)
		forw.SetOrigSym(t.OrigSym())
	}

	var newt *types.Type

	switch t.Kind() {
	case types.TTYPEPARAM:
		if t.Sym() == newsym && !targsChanged {
			// The substitution did not change the type.
			return t
		}
		// Substitute the underlying typeparam (e.g. T in P[T], see
		// the example describing type P[T] above).
		newt = ts.typ1(t.Underlying())
		assert(newt != t)

	case types.TARRAY:
		elem := t.Elem()
		newelem := ts.typ1(elem)
		if newelem != elem || targsChanged {
			newt = types.NewArray(newelem, t.NumElem())
		}

	case types.TPTR:
		elem := t.Elem()
		newelem := ts.typ1(elem)
		if newelem != elem || targsChanged {
			newt = types.NewPtr(newelem)
		}

	case types.TSLICE:
		elem := t.Elem()
		newelem := ts.typ1(elem)
		if newelem != elem || targsChanged {
			newt = types.NewSlice(newelem)
		}

	case types.TSTRUCT:
		newt = ts.tstruct(t, targsChanged)
		if newt == t {
			newt = nil
		}

	case types.TFUNC:
		newrecvs := ts.tstruct(t.Recvs(), false)
		newparams := ts.tstruct(t.Params(), false)
		newresults := ts.tstruct(t.Results(), false)
		// Translate the tparams of a signature.
		newtparams := ts.tstruct(t.TParams(), false)
		if newrecvs != t.Recvs() || newparams != t.Params() ||
			newresults != t.Results() || newtparams != t.TParams() || targsChanged {
			// If any types have changed, then the all the fields of
			// of recv, params, and results must be copied, because they have
			// offset fields that are dependent, and so must have an
			// independent copy for each new signature.
			var newrecv *types.Field
			if newrecvs.NumFields() > 0 {
				if newrecvs == t.Recvs() {
					newrecvs = ts.tstruct(t.Recvs(), true)
				}
				newrecv = newrecvs.Field(0)
			}
			if newparams == t.Params() {
				newparams = ts.tstruct(t.Params(), true)
			}
			if newresults == t.Results() {
				newresults = ts.tstruct(t.Results(), true)
			}
			var tparamfields []*types.Field
			if newtparams.HasTParam() {
				tparamfields = newtparams.FieldSlice()
			} else {
				// Completely remove the tparams from the resulting
				// signature, if the tparams are now concrete types.
				tparamfields = nil
			}
			newt = types.NewSignature(t.Pkg(), newrecv, tparamfields,
				newparams.FieldSlice(), newresults.FieldSlice())
		}

	case types.TINTER:
		newt = ts.tinter(t, targsChanged)
		if newt == t {
			newt = nil
		}

	case types.TMAP:
		newkey := ts.typ1(t.Key())
		newval := ts.typ1(t.Elem())
		if newkey != t.Key() || newval != t.Elem() || targsChanged {
			newt = types.NewMap(newkey, newval)
		}

	case types.TCHAN:
		elem := t.Elem()
		newelem := ts.typ1(elem)
		if newelem != elem || targsChanged {
			newt = types.NewChan(newelem, t.ChanDir())
		}
	case types.TFORW:
		if ts.SubstForwFunc != nil {
			return ts.SubstForwFunc(forw)
		} else {
			assert(false)
		}
	case types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64,
		types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64,
		types.TUINTPTR, types.TBOOL, types.TSTRING, types.TFLOAT32, types.TFLOAT64, types.TCOMPLEX64, types.TCOMPLEX128, types.TUNSAFEPTR:
		newt = t.Underlying()
	case types.TUNION:
		nt := t.NumTerms()
		newterms := make([]*types.Type, nt)
		tildes := make([]bool, nt)
		changed := false
		for i := 0; i < nt; i++ {
			term, tilde := t.Term(i)
			tildes[i] = tilde
			newterms[i] = ts.typ1(term)
			if newterms[i] != term {
				changed = true
			}
		}
		if changed {
			newt = types.NewUnion(newterms, tildes)
		}
	default:
		panic(fmt.Sprintf("Bad type in (*TSubster).Typ: %v", t.Kind()))
	}
	if newt == nil {
		// Even though there were typeparams in the type, there may be no
		// change if this is a function type for a function call (which will
		// have its own tparams/targs in the function instantiation).
		return t
	}

	if forw != nil {
		forw.SetUnderlying(newt)
		newt = forw
	}

	if !newt.HasTParam() && !newt.IsFuncArgStruct() {
		// Calculate the size of any new types created. These will be
		// deferred until the top-level ts.Typ() or g.typ() (if this is
		// called from g.fillinMethods()).
		types.CheckSize(newt)
	}

	if t.Kind() != types.TINTER && t.Methods().Len() > 0 {
		// Fill in the method info for the new type.
		var newfields []*types.Field
		newfields = make([]*types.Field, t.Methods().Len())
		for i, f := range t.Methods().Slice() {
			t2 := ts.typ1(f.Type)
			oldsym := f.Nname.Sym()

			// Use the name of the substituted receiver to create the
			// method name, since the receiver name may have many levels
			// of nesting (brackets) with type names to be substituted.
			recvType := t2.Recv().Type
			var nm string
			if recvType.IsPtr() {
				recvType = recvType.Elem()
				nm = "(*" + recvType.Sym().Name + ")." + f.Sym.Name
			} else {
				nm = recvType.Sym().Name + "." + f.Sym.Name
			}
			if recvType.RParams()[0].HasShape() {
				// We add "nofunc" to methods of shape type to avoid
				// conflict with the name of the shape-based helper
				// function. See header comment of MakeFuncInstSym.
				nm = "nofunc." + nm
			}
			newsym := oldsym.Pkg.Lookup(nm)
			var nname *ir.Name
			if newsym.Def != nil {
				nname = newsym.Def.(*ir.Name)
			} else {
				nname = ir.NewNameAt(f.Pos, newsym)
				nname.SetType(t2)
				ir.MarkFunc(nname)
				newsym.Def = nname
			}
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
			newfields[i].Nname = nname
		}
		newt.Methods().Set(newfields)
		if !newt.HasTParam() && !newt.HasShape() {
			// Generate all the methods for a new fully-instantiated type.

			NeedInstType(newt)
		}
	}
	return newt
}

// tstruct substitutes type params in types of the fields of a structure type. For
// each field, tstruct copies the Nname, and translates it if Nname is in
// ts.vars. To always force the creation of a new (top-level) struct,
// regardless of whether anything changed with the types or names of the struct's
// fields, set force to true.
func (ts *Tsubster) tstruct(t *types.Type, force bool) *types.Type {
	if t.NumFields() == 0 {
		if t.HasTParam() || t.HasShape() {
			// For an empty struct, we need to return a new type, if
			// substituting from a generic type or shape type, since it
			// will change HasTParam/HasShape flags.
			return types.NewStruct(t.Pkg(), nil)
		}
		return t
	}
	var newfields []*types.Field
	if force {
		newfields = make([]*types.Field, t.NumFields())
	}
	for i, f := range t.Fields().Slice() {
		t2 := ts.typ1(f.Type)
		if (t2 != f.Type || f.Nname != nil) && newfields == nil {
			newfields = make([]*types.Field, t.NumFields())
			for j := 0; j < i; j++ {
				newfields[j] = t.Field(j)
			}
		}
		if newfields != nil {
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
			newfields[i].Embedded = f.Embedded
			newfields[i].Note = f.Note
			if f.IsDDD() {
				newfields[i].SetIsDDD(true)
			}
			if f.Nointerface() {
				newfields[i].SetNointerface(true)
			}
			if f.Nname != nil && ts.Vars != nil {
				v := ts.Vars[f.Nname.(*ir.Name)]
				if v != nil {
					// This is the case where we are
					// translating the type of the function we
					// are substituting, so its dcls are in
					// the subst.ts.vars table, and we want to
					// change to reference the new dcl.
					newfields[i].Nname = v
				} else {
					// This is the case where we are
					// translating the type of a function
					// reference inside the function we are
					// substituting, so we leave the Nname
					// value as is.
					newfields[i].Nname = f.Nname
				}
			}
		}
	}
	if newfields != nil {
		news := types.NewStruct(t.Pkg(), newfields)
		news.StructType().Funarg = t.StructType().Funarg
		return news
	}
	return t

}

// tinter substitutes type params in types of the methods of an interface type.
func (ts *Tsubster) tinter(t *types.Type, force bool) *types.Type {
	if t.Methods().Len() == 0 {
		if t.HasTParam() || t.HasShape() {
			// For an empty interface, we need to return a new type, if
			// substituting from a generic type or shape type, since
			// since it will change HasTParam/HasShape flags.
			return types.NewInterface(t.Pkg(), nil, false)
		}
		return t
	}
	var newfields []*types.Field
	if force {
		newfields = make([]*types.Field, t.Methods().Len())
	}
	for i, f := range t.Methods().Slice() {
		t2 := ts.typ1(f.Type)
		if (t2 != f.Type || f.Nname != nil) && newfields == nil {
			newfields = make([]*types.Field, t.Methods().Len())
			for j := 0; j < i; j++ {
				newfields[j] = t.Methods().Index(j)
			}
		}
		if newfields != nil {
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
		}
	}
	if newfields != nil {
		return types.NewInterface(t.Pkg(), newfields, t.IsImplicit())
	}
	return t
}

// genericSym returns the name of the base generic type for the type named by
// sym. It simply returns the name obtained by removing everything after the
// first bracket ("[").
func genericTypeName(sym *types.Sym) string {
	return sym.Name[0:strings.Index(sym.Name, "[")]
}

// getShapes appends the list of the shape types that are used within type t to
// listp. The type traversal is simplified for two reasons: (1) we can always stop a
// type traversal when t.HasShape() is false; and (2) shape types can't appear inside
// a named type, except for the type args of a generic type. So, the traversal will
// always stop before we have to deal with recursive types.
func getShapes(t *types.Type, listp *[]*types.Type) {
	if !t.HasShape() {
		return
	}
	if t.IsShape() {
		*listp = append(*listp, t)
		return
	}

	if t.Sym() != nil {
		// A named type can't have shapes in it, except for type args of a
		// generic type. We will have to deal with this differently once we
		// alloc local types in generic functions (#47631).
		for _, rparam := range t.RParams() {
			getShapes(rparam, listp)
		}
		return
	}

	switch t.Kind() {
	case types.TARRAY, types.TPTR, types.TSLICE, types.TCHAN:
		getShapes(t.Elem(), listp)

	case types.TSTRUCT:
		for _, f := range t.FieldSlice() {
			getShapes(f.Type, listp)
		}

	case types.TFUNC:
		for _, f := range t.Recvs().FieldSlice() {
			getShapes(f.Type, listp)
		}
		for _, f := range t.Params().FieldSlice() {
			getShapes(f.Type, listp)
		}
		for _, f := range t.Results().FieldSlice() {
			getShapes(f.Type, listp)
		}
		for _, f := range t.TParams().FieldSlice() {
			getShapes(f.Type, listp)
		}

	case types.TINTER:
		for _, f := range t.Methods().Slice() {
			getShapes(f.Type, listp)
		}

	case types.TMAP:
		getShapes(t.Key(), listp)
		getShapes(t.Elem(), listp)

	default:
		panic(fmt.Sprintf("Bad type in getShapes: %v", t.Kind()))
	}

}

// Shapify takes a concrete type and a type param index, and returns a GCshape type that can
// be used in place of the input type and still generate identical code.
// No methods are added - all methods calls directly on a shape should
// be done by converting to an interface using the dictionary.
//
// For now, we only consider two types to have the same shape, if they have exactly
// the same underlying type or they are both pointer types.
//
//  tparam is the associated typeparam - it must be TTYPEPARAM type. If there is a
//  structural type for the associated type param (not common), then a pointer type t
//  is mapped to its underlying type, rather than being merged with other pointers.
//
//  Shape types are also distinguished by the index of the type in a type param/arg
//  list. We need to do this so we can distinguish and substitute properly for two
//  type params in the same function that have the same shape for a particular
//  instantiation.
func Shapify(t *types.Type, index int, tparam *types.Type) *types.Type {
	assert(!t.IsShape())
	if t.HasShape() {
		// We are sometimes dealing with types from a shape instantiation
		// that were constructed from existing shape types, so t may
		// sometimes have shape types inside it. In that case, we find all
		// those shape types with getShapes() and replace them with their
		// underlying type.
		//
		// If we don't do this, we may create extra unneeded shape types that
		// have these other shape types embedded in them. This may lead to
		// generating extra shape instantiations, and a mismatch between the
		// instantiations that we used in generating dictionaries and the
		// instantations that are actually called. (#51303).
		list := []*types.Type{}
		getShapes(t, &list)
		list2 := make([]*types.Type, len(list))
		for i, shape := range list {
			list2[i] = shape.Underlying()
		}
		ts := Tsubster{
			Tparams: list,
			Targs:   list2,
		}
		t = ts.Typ(t)
	}
	// Map all types with the same underlying type to the same shape.
	u := t.Underlying()

	// All pointers have the same shape.
	// TODO: Make unsafe.Pointer the same shape as normal pointers.
	// Note: pointers to arrays are special because of slice-to-array-pointer
	// conversions. See issue 49295.
	if u.Kind() == types.TPTR && u.Elem().Kind() != types.TARRAY &&
		tparam.Bound().StructuralType() == nil {
		u = types.Types[types.TUINT8].PtrTo()
	}

	if shapeMap == nil {
		shapeMap = map[int]map[*types.Type]*types.Type{}
	}
	submap := shapeMap[index]
	if submap == nil {
		submap = map[*types.Type]*types.Type{}
		shapeMap[index] = submap
	}
	if s := submap[u]; s != nil {
		return s
	}

	// LinkString specifies the type uniquely, but has no spaces.
	nm := fmt.Sprintf("%s_%d", u.LinkString(), index)
	sym := types.ShapePkg.Lookup(nm)
	if sym.Def != nil {
		// Use any existing type with the same name
		submap[u] = sym.Def.Type()
		return submap[u]
	}
	name := ir.NewDeclNameAt(u.Pos(), ir.OTYPE, sym)
	s := types.NewNamed(name)
	sym.Def = name
	s.SetUnderlying(u)
	s.SetIsShape(true)
	s.SetHasShape(true)
	types.CalcSize(s)
	name.SetType(s)
	name.SetTypecheck(1)
	submap[u] = s
	return s
}

var shapeMap map[int]map[*types.Type]*types.Type
