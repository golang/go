// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"os"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

type Error struct {
	pos src.XPos
	msg string
}

var errors []Error

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

func errorexit() {
	flusherrors()
	if outfile != "" {
		os.Remove(outfile)
	}
	os.Exit(2)
}

func adderrorname(n *Node) {
	if n.Op != ODOT {
		return
	}
	old := fmt.Sprintf("%v: undefined: %v\n", n.Line(), n.Left)
	if len(errors) > 0 && errors[len(errors)-1].pos.Line() == n.Pos.Line() && errors[len(errors)-1].msg == old {
		errors[len(errors)-1].msg = fmt.Sprintf("%v: undefined: %v in %v\n", n.Line(), n.Left, n)
	}
}

func adderr(pos src.XPos, format string, args ...interface{}) {
	errors = append(errors, Error{
		pos: pos,
		msg: fmt.Sprintf("%v: %s\n", linestr(pos), fmt.Sprintf(format, args...)),
	})
}

// byPos sorts errors by source position.
type byPos []Error

func (x byPos) Len() int           { return len(x) }
func (x byPos) Less(i, j int) bool { return x[i].pos.Before(x[j].pos) }
func (x byPos) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }

// flusherrors sorts errors seen so far by line number, prints them to stdout,
// and empties the errors array.
func flusherrors() {
	Ctxt.Bso.Flush()
	if len(errors) == 0 {
		return
	}
	sort.Stable(byPos(errors))
	for i, err := range errors {
		if i == 0 || err.msg != errors[i-1].msg {
			fmt.Printf("%s", err.msg)
		}
	}
	errors = errors[:0]
}

func hcrash() {
	if Debug['h'] != 0 {
		flusherrors()
		if outfile != "" {
			os.Remove(outfile)
		}
		var x *int
		*x = 0
	}
}

func linestr(pos src.XPos) string {
	return Ctxt.OutermostPos(pos).Format(Debug['C'] == 0, Debug['L'] == 1)
}

// lasterror keeps track of the most recently issued error.
// It is used to avoid multiple error messages on the same
// line.
var lasterror struct {
	syntax src.XPos // source position of last syntax error
	other  src.XPos // source position of last non-syntax error
	msg    string   // error message of last non-syntax error
}

// sameline reports whether two positions a, b are on the same line.
func sameline(a, b src.XPos) bool {
	p := Ctxt.PosTable.Pos(a)
	q := Ctxt.PosTable.Pos(b)
	return p.Base() == q.Base() && p.Line() == q.Line()
}

func yyerrorl(pos src.XPos, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)

	if strings.HasPrefix(msg, "syntax error") {
		nsyntaxerrors++
		// only one syntax error per line, no matter what error
		if sameline(lasterror.syntax, pos) {
			return
		}
		lasterror.syntax = pos
	} else {
		// only one of multiple equal non-syntax errors per line
		// (flusherrors shows only one of them, so we filter them
		// here as best as we can (they may not appear in order)
		// so that we don't count them here and exit early, and
		// then have nothing to show for.)
		if sameline(lasterror.other, pos) && lasterror.msg == msg {
			return
		}
		lasterror.other = pos
		lasterror.msg = msg
	}

	adderr(pos, "%s", msg)

	hcrash()
	nerrors++
	if nsavederrors+nerrors >= 10 && Debug['e'] == 0 {
		flusherrors()
		fmt.Printf("%v: too many errors\n", linestr(pos))
		errorexit()
	}
}

func yyerror(format string, args ...interface{}) {
	yyerrorl(lineno, format, args...)
}

func Warn(fmt_ string, args ...interface{}) {
	adderr(lineno, fmt_, args...)

	hcrash()
}

func Warnl(line src.XPos, fmt_ string, args ...interface{}) {
	adderr(line, fmt_, args...)
	if Debug['m'] != 0 {
		flusherrors()
	}
}

func Fatalf(fmt_ string, args ...interface{}) {
	flusherrors()

	if Debug_panic != 0 || nsavederrors+nerrors == 0 {
		fmt.Printf("%v: internal compiler error: ", linestr(lineno))
		fmt.Printf(fmt_, args...)
		fmt.Printf("\n")

		// If this is a released compiler version, ask for a bug report.
		if strings.HasPrefix(objabi.Version, "go") {
			fmt.Printf("\n")
			fmt.Printf("Please file a bug report including a short program that triggers the error.\n")
			fmt.Printf("https://golang.org/issue/new\n")
		} else {
			// Not a release; dump a stack trace, too.
			fmt.Println()
			os.Stdout.Write(debug.Stack())
			fmt.Println()
		}
	}

	hcrash()
	errorexit()
}

func setlineno(n *Node) src.XPos {
	lno := lineno
	if n != nil {
		switch n.Op {
		case ONAME, OPACK:
			break

		case OLITERAL, OTYPE:
			if n.Sym != nil {
				break
			}
			fallthrough

		default:
			lineno = n.Pos
			if !lineno.IsKnown() {
				if Debug['K'] != 0 {
					Warn("setlineno: unknown position (line 0)")
				}
				lineno = lno
			}
		}
	}

	return lno
}

func lookup(name string) *types.Sym {
	return localpkg.Lookup(name)
}

// lookupN looks up the symbol starting with prefix and ending with
// the decimal n. If prefix is too long, lookupN panics.
func lookupN(prefix string, n int) *types.Sym {
	var buf [20]byte // plenty long enough for all current users
	copy(buf[:], prefix)
	b := strconv.AppendInt(buf[:len(prefix)], int64(n), 10)
	return localpkg.LookupBytes(b)
}

// autolabel generates a new Name node for use with
// an automatically generated label.
// prefix is a short mnemonic (e.g. ".s" for switch)
// to help with debugging.
// It should begin with "." to avoid conflicts with
// user labels.
func autolabel(prefix string) *types.Sym {
	if prefix[0] != '.' {
		Fatalf("autolabel prefix must start with '.', have %q", prefix)
	}
	fn := Curfn
	if Curfn == nil {
		Fatalf("autolabel outside function")
	}
	n := fn.Func.Label
	fn.Func.Label++
	return lookupN(prefix, int(n))
}

func restrictlookup(name string, pkg *types.Pkg) *types.Sym {
	if !types.IsExported(name) && pkg != localpkg {
		yyerror("cannot refer to unexported name %s.%s", pkg.Name, name)
	}
	return pkg.Lookup(name)
}

// find all the exported symbols in package opkg
// and make them available in the current package
func importdot(opkg *types.Pkg, pack *Node) {
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
			redeclare(lineno, s1, pkgerror)
			continue
		}

		s1.Def = s.Def
		s1.Block = s.Block
		if asNode(s1.Def).Name == nil {
			Dump("s1def", asNode(s1.Def))
			Fatalf("missing Name")
		}
		asNode(s1.Def).Name.Pack = pack
		s1.Origpkg = opkg
		n++
	}

	if n == 0 {
		// can't possibly be used - there were no symbols
		yyerrorl(pack.Pos, "imported and not used: %q", opkg.Path)
	}
}

func nod(op Op, nleft, nright *Node) *Node {
	return nodl(lineno, op, nleft, nright)
}

func nodl(pos src.XPos, op Op, nleft, nright *Node) *Node {
	var n *Node
	switch op {
	case OCLOSURE, ODCLFUNC:
		var x struct {
			Node
			Func
		}
		n = &x.Node
		n.Func = &x.Func
	case ONAME:
		Fatalf("use newname instead")
	case OLABEL, OPACK:
		var x struct {
			Node
			Name
		}
		n = &x.Node
		n.Name = &x.Name
	default:
		n = new(Node)
	}
	n.Op = op
	n.Left = nleft
	n.Right = nright
	n.Pos = pos
	n.Xoffset = BADWIDTH
	n.Orig = n
	return n
}

// newname returns a new ONAME Node associated with symbol s.
func newname(s *types.Sym) *Node {
	n := newnamel(lineno, s)
	n.Name.Curfn = Curfn
	return n
}

// newname returns a new ONAME Node associated with symbol s at position pos.
// The caller is responsible for setting n.Name.Curfn.
func newnamel(pos src.XPos, s *types.Sym) *Node {
	if s == nil {
		Fatalf("newnamel nil")
	}

	var x struct {
		Node
		Name
		Param
	}
	n := &x.Node
	n.Name = &x.Name
	n.Name.Param = &x.Param

	n.Op = ONAME
	n.Pos = pos
	n.Orig = n

	n.Sym = s
	n.SetAddable(true)
	return n
}

// nodSym makes a Node with Op op and with the Left field set to left
// and the Sym field set to sym. This is for ODOT and friends.
func nodSym(op Op, left *Node, sym *types.Sym) *Node {
	n := nod(op, left, nil)
	n.Sym = sym
	return n
}

// rawcopy returns a shallow copy of n.
// Note: copy or sepcopy (rather than rawcopy) is usually the
//       correct choice (see comment with Node.copy, below).
func (n *Node) rawcopy() *Node {
	copy := *n
	return &copy
}

// sepcopy returns a separate shallow copy of n, with the copy's
// Orig pointing to itself.
func (n *Node) sepcopy() *Node {
	copy := *n
	copy.Orig = &copy
	return &copy
}

// copy returns shallow copy of n and adjusts the copy's Orig if
// necessary: In general, if n.Orig points to itself, the copy's
// Orig should point to itself as well. Otherwise, if n is modified,
// the copy's Orig node appears modified, too, and then doesn't
// represent the original node anymore.
// (This caused the wrong complit Op to be used when printing error
// messages; see issues #26855, #27765).
func (n *Node) copy() *Node {
	copy := *n
	if n.Orig == n {
		copy.Orig = &copy
	}
	return &copy
}

// methcmp sorts methods by symbol.
type methcmp []*types.Field

func (x methcmp) Len() int           { return len(x) }
func (x methcmp) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x methcmp) Less(i, j int) bool { return x[i].Sym.Less(x[j].Sym) }

func nodintconst(v int64) *Node {
	u := new(Mpint)
	u.SetInt64(v)
	return nodlit(Val{u})
}

func nodfltconst(v *Mpflt) *Node {
	u := newMpflt()
	u.Set(v)
	return nodlit(Val{u})
}

func nodnil() *Node {
	return nodlit(Val{new(NilVal)})
}

func nodbool(b bool) *Node {
	return nodlit(Val{b})
}

func nodstr(s string) *Node {
	return nodlit(Val{s})
}

// treecopy recursively copies n, with the exception of
// ONAME, OLITERAL, OTYPE, and non-iota ONONAME leaves.
// Copies of iota ONONAME nodes are assigned the current
// value of iota_. If pos.IsKnown(), it sets the source
// position of newly allocated nodes to pos.
func treecopy(n *Node, pos src.XPos) *Node {
	if n == nil {
		return nil
	}

	switch n.Op {
	default:
		m := n.sepcopy()
		m.Left = treecopy(n.Left, pos)
		m.Right = treecopy(n.Right, pos)
		m.List.Set(listtreecopy(n.List.Slice(), pos))
		if pos.IsKnown() {
			m.Pos = pos
		}
		if m.Name != nil && n.Op != ODCLFIELD {
			Dump("treecopy", n)
			Fatalf("treecopy Name")
		}
		return m

	case OPACK:
		// OPACK nodes are never valid in const value declarations,
		// but allow them like any other declared symbol to avoid
		// crashing (golang.org/issue/11361).
		fallthrough

	case ONAME, ONONAME, OLITERAL, OTYPE:
		return n

	}
}

// isNil reports whether n represents the universal untyped zero value "nil".
func (n *Node) isNil() bool {
	// Check n.Orig because constant propagation may produce typed nil constants,
	// which don't exist in the Go spec.
	return Isconst(n.Orig, CTNIL)
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

func (n *Node) isBlank() bool {
	if n == nil {
		return false
	}
	return n.Sym.IsBlank()
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
	case TARRAY, TCHAN, TFUNC, TMAP, TSLICE, TSTRING, TSTRUCT:
		return t
	}
	return nil
}

// Are t1 and t2 equal struct types when field names are ignored?
// For deciding whether the result struct from g can be copied
// directly when compiling f(g()).
func eqtypenoname(t1 *types.Type, t2 *types.Type) bool {
	if t1 == nil || t2 == nil || !t1.IsStruct() || !t2.IsStruct() {
		return false
	}

	if t1.NumFields() != t2.NumFields() {
		return false
	}
	for i, f1 := range t1.FieldSlice() {
		f2 := t2.Field(i)
		if !types.Identical(f1.Type, f2.Type) {
			return false
		}
	}
	return true
}

// Is type src assignment compatible to type dst?
// If so, return op code to use in conversion.
// If not, return 0.
func assignop(src *types.Type, dst *types.Type, why *string) Op {
	if why != nil {
		*why = ""
	}

	if src == dst {
		return OCONVNOP
	}
	if src == nil || dst == nil || src.Etype == TFORW || dst.Etype == TFORW || src.Orig == nil || dst.Orig == nil {
		return 0
	}

	// 1. src type is identical to dst.
	if types.Identical(src, dst) {
		return OCONVNOP
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
			return OCONVNOP
		}
		if (src.Sym == nil || dst.Sym == nil) && !src.IsInterface() {
			// Conversion between two types, at least one unnamed,
			// needs no conversion. The exception is nonempty interfaces
			// which need to have their itab updated.
			return OCONVNOP
		}
	}

	// 3. dst is an interface type and src implements dst.
	if dst.IsInterface() && src.Etype != TNIL {
		var missing, have *types.Field
		var ptr int
		if implements(src, dst, &missing, &have, &ptr) {
			return OCONVIFACE
		}

		// we'll have complained about this method anyway, suppress spurious messages.
		if have != nil && have.Sym == missing.Sym && (have.Type.Broke() || missing.Type.Broke()) {
			return OCONVIFACE
		}

		if why != nil {
			if isptrto(src, TINTER) {
				*why = fmt.Sprintf(":\n\t%v is pointer to interface, not interface", src)
			} else if have != nil && have.Sym == missing.Sym && have.Nointerface() {
				*why = fmt.Sprintf(":\n\t%v does not implement %v (%v method is marked 'nointerface')", src, dst, missing.Sym)
			} else if have != nil && have.Sym == missing.Sym {
				*why = fmt.Sprintf(":\n\t%v does not implement %v (wrong type for %v method)\n"+
					"\t\thave %v%0S\n\t\twant %v%0S", src, dst, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
			} else if ptr != 0 {
				*why = fmt.Sprintf(":\n\t%v does not implement %v (%v method has pointer receiver)", src, dst, missing.Sym)
			} else if have != nil {
				*why = fmt.Sprintf(":\n\t%v does not implement %v (missing %v method)\n"+
					"\t\thave %v%0S\n\t\twant %v%0S", src, dst, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
			} else {
				*why = fmt.Sprintf(":\n\t%v does not implement %v (missing %v method)", src, dst, missing.Sym)
			}
		}

		return 0
	}

	if isptrto(dst, TINTER) {
		if why != nil {
			*why = fmt.Sprintf(":\n\t%v is pointer to interface, not interface", dst)
		}
		return 0
	}

	if src.IsInterface() && dst.Etype != TBLANK {
		var missing, have *types.Field
		var ptr int
		if why != nil && implements(dst, src, &missing, &have, &ptr) {
			*why = ": need type assertion"
		}
		return 0
	}

	// 4. src is a bidirectional channel value, dst is a channel type,
	// src and dst have identical element types, and
	// either src or dst is not a named type.
	if src.IsChan() && src.ChanDir() == types.Cboth && dst.IsChan() {
		if types.Identical(src.Elem(), dst.Elem()) && (src.Sym == nil || dst.Sym == nil) {
			return OCONVNOP
		}
	}

	// 5. src is the predeclared identifier nil and dst is a nillable type.
	if src.Etype == TNIL {
		switch dst.Etype {
		case TPTR,
			TFUNC,
			TMAP,
			TCHAN,
			TINTER,
			TSLICE:
			return OCONVNOP
		}
	}

	// 6. rule about untyped constants - already converted by defaultlit.

	// 7. Any typed value can be assigned to the blank identifier.
	if dst.Etype == TBLANK {
		return OCONVNOP
	}

	return 0
}

// Can we convert a value of type src to a value of type dst?
// If so, return op code to use in conversion (maybe OCONVNOP).
// If not, return 0.
func convertop(src *types.Type, dst *types.Type, why *string) Op {
	if why != nil {
		*why = ""
	}

	if src == dst {
		return OCONVNOP
	}
	if src == nil || dst == nil {
		return 0
	}

	// Conversions from regular to go:notinheap are not allowed
	// (unless it's unsafe.Pointer). These are runtime-specific
	// rules.
	// (a) Disallow (*T) to (*U) where T is go:notinheap but U isn't.
	if src.IsPtr() && dst.IsPtr() && dst.Elem().NotInHeap() && !src.Elem().NotInHeap() {
		if why != nil {
			*why = fmt.Sprintf(":\n\t%v is go:notinheap, but %v is not", dst.Elem(), src.Elem())
		}
		return 0
	}
	// (b) Disallow string to []T where T is go:notinheap.
	if src.IsString() && dst.IsSlice() && dst.Elem().NotInHeap() && (dst.Elem().Etype == types.Bytetype.Etype || dst.Elem().Etype == types.Runetype.Etype) {
		if why != nil {
			*why = fmt.Sprintf(":\n\t%v is go:notinheap", dst.Elem())
		}
		return 0
	}

	// 1. src can be assigned to dst.
	op := assignop(src, dst, why)
	if op != 0 {
		return op
	}

	// The rules for interfaces are no different in conversions
	// than assignments. If interfaces are involved, stop now
	// with the good message from assignop.
	// Otherwise clear the error.
	if src.IsInterface() || dst.IsInterface() {
		return 0
	}
	if why != nil {
		*why = ""
	}

	// 2. Ignoring struct tags, src and dst have identical underlying types.
	if types.IdenticalIgnoreTags(src.Orig, dst.Orig) {
		return OCONVNOP
	}

	// 3. src and dst are unnamed pointer types and, ignoring struct tags,
	// their base types have identical underlying types.
	if src.IsPtr() && dst.IsPtr() && src.Sym == nil && dst.Sym == nil {
		if types.IdenticalIgnoreTags(src.Elem().Orig, dst.Elem().Orig) {
			return OCONVNOP
		}
	}

	// 4. src and dst are both integer or floating point types.
	if (src.IsInteger() || src.IsFloat()) && (dst.IsInteger() || dst.IsFloat()) {
		if simtype[src.Etype] == simtype[dst.Etype] {
			return OCONVNOP
		}
		return OCONV
	}

	// 5. src and dst are both complex types.
	if src.IsComplex() && dst.IsComplex() {
		if simtype[src.Etype] == simtype[dst.Etype] {
			return OCONVNOP
		}
		return OCONV
	}

	// 6. src is an integer or has type []byte or []rune
	// and dst is a string type.
	if src.IsInteger() && dst.IsString() {
		return ORUNESTR
	}

	if src.IsSlice() && dst.IsString() {
		if src.Elem().Etype == types.Bytetype.Etype {
			return OBYTES2STR
		}
		if src.Elem().Etype == types.Runetype.Etype {
			return ORUNES2STR
		}
	}

	// 7. src is a string and dst is []byte or []rune.
	// String to slice.
	if src.IsString() && dst.IsSlice() {
		if dst.Elem().Etype == types.Bytetype.Etype {
			return OSTR2BYTES
		}
		if dst.Elem().Etype == types.Runetype.Etype {
			return OSTR2RUNES
		}
	}

	// 8. src is a pointer or uintptr and dst is unsafe.Pointer.
	if (src.IsPtr() || src.Etype == TUINTPTR) && dst.Etype == TUNSAFEPTR {
		return OCONVNOP
	}

	// 9. src is unsafe.Pointer and dst is a pointer or uintptr.
	if src.Etype == TUNSAFEPTR && (dst.IsPtr() || dst.Etype == TUINTPTR) {
		return OCONVNOP
	}

	// src is map and dst is a pointer to corresponding hmap.
	// This rule is needed for the implementation detail that
	// go gc maps are implemented as a pointer to a hmap struct.
	if src.Etype == TMAP && dst.IsPtr() &&
		src.MapType().Hmap == dst.Elem() {
		return OCONVNOP
	}

	return 0
}

func assignconv(n *Node, t *types.Type, context string) *Node {
	return assignconvfn(n, t, func() string { return context })
}

// Convert node n for assignment to type t.
func assignconvfn(n *Node, t *types.Type, context func() string) *Node {
	if n == nil || n.Type == nil || n.Type.Broke() {
		return n
	}

	if t.Etype == TBLANK && n.Type.Etype == TNIL {
		yyerror("use of untyped nil")
	}

	old := n
	od := old.Diag()
	old.SetDiag(true) // silence errors about n; we'll issue one below
	n = defaultlit(n, t)
	old.SetDiag(od)
	if t.Etype == TBLANK {
		return n
	}

	// Convert ideal bool from comparison to plain bool
	// if the next step is non-bool (like interface{}).
	if n.Type == types.Idealbool && !t.IsBoolean() {
		if n.Op == ONAME || n.Op == OLITERAL {
			r := nod(OCONVNOP, n, nil)
			r.Type = types.Types[TBOOL]
			r.SetTypecheck(1)
			r.SetImplicit(true)
			n = r
		}
	}

	if types.Identical(n.Type, t) {
		return n
	}

	var why string
	op := assignop(n.Type, t, &why)
	if op == 0 {
		if !old.Diag() {
			yyerror("cannot use %L as type %v in %s%s", n, t, context(), why)
		}
		op = OCONV
	}

	r := nod(op, n, nil)
	r.Type = t
	r.SetTypecheck(1)
	r.SetImplicit(true)
	r.Orig = n.Orig
	return r
}

// IsMethod reports whether n is a method.
// n must be a function or a method.
func (n *Node) IsMethod() bool {
	return n.Type.Recv() != nil
}

// SliceBounds returns n's slice bounds: low, high, and max in expr[low:high:max].
// n must be a slice expression. max is nil if n is a simple slice expression.
func (n *Node) SliceBounds() (low, high, max *Node) {
	if n.List.Len() == 0 {
		return nil, nil, nil
	}

	switch n.Op {
	case OSLICE, OSLICEARR, OSLICESTR:
		s := n.List.Slice()
		return s[0], s[1], nil
	case OSLICE3, OSLICE3ARR:
		s := n.List.Slice()
		return s[0], s[1], s[2]
	}
	Fatalf("SliceBounds op %v: %v", n.Op, n)
	return nil, nil, nil
}

// SetSliceBounds sets n's slice bounds, where n is a slice expression.
// n must be a slice expression. If max is non-nil, n must be a full slice expression.
func (n *Node) SetSliceBounds(low, high, max *Node) {
	switch n.Op {
	case OSLICE, OSLICEARR, OSLICESTR:
		if max != nil {
			Fatalf("SetSliceBounds %v given three bounds", n.Op)
		}
		s := n.List.Slice()
		if s == nil {
			if low == nil && high == nil {
				return
			}
			n.List.Set2(low, high)
			return
		}
		s[0] = low
		s[1] = high
		return
	case OSLICE3, OSLICE3ARR:
		s := n.List.Slice()
		if s == nil {
			if low == nil && high == nil && max == nil {
				return
			}
			n.List.Set3(low, high, max)
			return
		}
		s[0] = low
		s[1] = high
		s[2] = max
		return
	}
	Fatalf("SetSliceBounds op %v: %v", n.Op, n)
}

// IsSlice3 reports whether o is a slice3 op (OSLICE3, OSLICE3ARR).
// o must be a slicing op.
func (o Op) IsSlice3() bool {
	switch o {
	case OSLICE, OSLICEARR, OSLICESTR:
		return false
	case OSLICE3, OSLICE3ARR:
		return true
	}
	Fatalf("IsSlice3 op %v", o)
	return false
}

// labeledControl returns the control flow Node (for, switch, select)
// associated with the label n, if any.
func (n *Node) labeledControl() *Node {
	if n.Op != OLABEL {
		Fatalf("labeledControl %v", n.Op)
	}
	ctl := n.Name.Defn
	if ctl == nil {
		return nil
	}
	switch ctl.Op {
	case OFOR, OFORUNTIL, OSWITCH, OSELECT:
		return ctl
	}
	return nil
}

func syslook(name string) *Node {
	s := Runtimepkg.Lookup(name)
	if s == nil || s.Def == nil {
		Fatalf("syslook: can't find runtime.%s", name)
	}
	return asNode(s.Def)
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
func updateHasCall(n *Node) {
	if n == nil {
		return
	}
	n.SetHasCall(calcHasCall(n))
}

func calcHasCall(n *Node) bool {
	if n.Ninit.Len() != 0 {
		// TODO(mdempsky): This seems overly conservative.
		return true
	}

	switch n.Op {
	case OLITERAL, ONAME, OTYPE:
		if n.HasCall() {
			Fatalf("OLITERAL/ONAME/OTYPE should never have calls: %+v", n)
		}
		return false
	case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER:
		return true
	case OANDAND, OOROR:
		// hard with instrumented code
		if instrumenting {
			return true
		}
	case OINDEX, OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR, OSLICESTR,
		ODEREF, ODOTPTR, ODOTTYPE, ODIV, OMOD:
		// These ops might panic, make sure they are done
		// before we start marshaling args for a call. See issue 16760.
		return true

	// When using soft-float, these ops might be rewritten to function calls
	// so we ensure they are evaluated first.
	case OADD, OSUB, ONEG, OMUL:
		if thearch.SoftFloat && (isFloat[n.Type.Etype] || isComplex[n.Type.Etype]) {
			return true
		}
	case OLT, OEQ, ONE, OLE, OGE, OGT:
		if thearch.SoftFloat && (isFloat[n.Left.Type.Etype] || isComplex[n.Left.Type.Etype]) {
			return true
		}
	case OCONV:
		if thearch.SoftFloat && ((isFloat[n.Type.Etype] || isComplex[n.Type.Etype]) || (isFloat[n.Left.Type.Etype] || isComplex[n.Left.Type.Etype])) {
			return true
		}
	}

	if n.Left != nil && n.Left.HasCall() {
		return true
	}
	if n.Right != nil && n.Right.HasCall() {
		return true
	}
	return false
}

func badtype(op Op, tl *types.Type, tr *types.Type) {
	fmt_ := ""
	if tl != nil {
		fmt_ += fmt.Sprintf("\n\t%v", tl)
	}
	if tr != nil {
		fmt_ += fmt.Sprintf("\n\t%v", tr)
	}

	// common mistake: *struct and *interface.
	if tl != nil && tr != nil && tl.IsPtr() && tr.IsPtr() {
		if tl.Elem().IsStruct() && tr.Elem().IsInterface() {
			fmt_ += "\n\t(*struct vs *interface)"
		} else if tl.Elem().IsInterface() && tr.Elem().IsStruct() {
			fmt_ += "\n\t(*interface vs *struct)"
		}
	}

	s := fmt_
	yyerror("illegal types for operand: %v%s", op, s)
}

// brcom returns !(op).
// For example, brcom(==) is !=.
func brcom(op Op) Op {
	switch op {
	case OEQ:
		return ONE
	case ONE:
		return OEQ
	case OLT:
		return OGE
	case OGT:
		return OLE
	case OLE:
		return OGT
	case OGE:
		return OLT
	}
	Fatalf("brcom: no com for %v\n", op)
	return op
}

// brrev returns reverse(op).
// For example, Brrev(<) is >.
func brrev(op Op) Op {
	switch op {
	case OEQ:
		return OEQ
	case ONE:
		return ONE
	case OLT:
		return OGT
	case OGT:
		return OLT
	case OLE:
		return OGE
	case OGE:
		return OLE
	}
	Fatalf("brrev: no rev for %v\n", op)
	return op
}

// return side effect-free n, appending side effects to init.
// result is assignable if n is.
func safeexpr(n *Node, init *Nodes) *Node {
	if n == nil {
		return nil
	}

	if n.Ninit.Len() != 0 {
		walkstmtlist(n.Ninit.Slice())
		init.AppendNodes(&n.Ninit)
	}

	switch n.Op {
	case ONAME, OLITERAL:
		return n

	case ODOT, OLEN, OCAP:
		l := safeexpr(n.Left, init)
		if l == n.Left {
			return n
		}
		r := n.copy()
		r.Left = l
		r = typecheck(r, ctxExpr)
		r = walkexpr(r, init)
		return r

	case ODOTPTR, ODEREF:
		l := safeexpr(n.Left, init)
		if l == n.Left {
			return n
		}
		a := n.copy()
		a.Left = l
		a = walkexpr(a, init)
		return a

	case OINDEX, OINDEXMAP:
		l := safeexpr(n.Left, init)
		r := safeexpr(n.Right, init)
		if l == n.Left && r == n.Right {
			return n
		}
		a := n.copy()
		a.Left = l
		a.Right = r
		a = walkexpr(a, init)
		return a

	case OSTRUCTLIT, OARRAYLIT, OSLICELIT:
		if isStaticCompositeLiteral(n) {
			return n
		}
	}

	// make a copy; must not be used as an lvalue
	if islvalue(n) {
		Fatalf("missing lvalue case in safeexpr: %v", n)
	}
	return cheapexpr(n, init)
}

func copyexpr(n *Node, t *types.Type, init *Nodes) *Node {
	l := temp(t)
	a := nod(OAS, l, n)
	a = typecheck(a, ctxStmt)
	a = walkexpr(a, init)
	init.Append(a)
	return l
}

// return side-effect free and cheap n, appending side effects to init.
// result may not be assignable.
func cheapexpr(n *Node, init *Nodes) *Node {
	switch n.Op {
	case ONAME, OLITERAL:
		return n
	}

	return copyexpr(n, n.Type, init)
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
			if f.Sym == s || (ignorecase && f.Type.Etype == TFUNC && f.Type.Recv() != nil && strings.EqualFold(f.Sym.Name, s.Name)) {
				if save != nil {
					*save = f
				}
				c++
			}
		}
	}

	u = methtype(t)
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
func adddot(n *Node) *Node {
	n.Left = typecheck(n.Left, Etype|ctxExpr)
	if n.Left.Diag() {
		n.SetDiag(true)
	}
	t := n.Left.Type
	if t == nil {
		return n
	}

	if n.Left.Op == OTYPE {
		return n
	}

	s := n.Sym
	if s == nil {
		return n
	}

	switch path, ambig := dotpath(s, t, nil, false); {
	case path != nil:
		// rebuild elided dots
		for c := len(path) - 1; c >= 0; c-- {
			n.Left = nodSym(ODOT, n.Left, path[c].field.Sym)
			n.Left.SetImplicit(true)
		}
	case ambig:
		yyerror("ambiguous selector %v", n)
		n.Left = nil
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
		if f.Type.Etype != TFUNC || f.Type.Recv() == nil {
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
func structargs(tl *types.Type, mustname bool) []*Node {
	var args []*Node
	gen := 0
	for _, t := range tl.Fields().Slice() {
		s := t.Sym
		if mustname && (s == nil || s.Name == "_") {
			// invent a name so that we can refer to it in the trampoline
			s = lookupN(".anon", gen)
			gen++
		}
		a := symfield(s, t.Type)
		a.Pos = t.Pos
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
	if false && Debug['r'] != 0 {
		fmt.Printf("genwrapper rcvrtype=%v method=%v newnam=%v\n", rcvr, method, newnam)
	}

	// Only generate (*T).M wrappers for T.M in T's own package.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type &&
		rcvr.Elem().Sym != nil && rcvr.Elem().Sym.Pkg != localpkg {
		return
	}

	// Only generate I.M wrappers for I in I's own package.
	if rcvr.IsInterface() && rcvr.Sym != nil && rcvr.Sym.Pkg != localpkg {
		return
	}

	lineno = autogeneratedPos
	dclcontext = PEXTERN

	tfn := nod(OTFUNC, nil, nil)
	tfn.Left = namedfield(".this", rcvr)
	tfn.List.Set(structargs(method.Type.Params(), true))
	tfn.Rlist.Set(structargs(method.Type.Results(), false))

	disableExport(newnam)
	fn := dclfunc(newnam, tfn)
	fn.Func.SetDupok(true)

	nthis := asNode(tfn.Type.Recv().Nname)

	methodrcvr := method.Type.Recv().Type

	// generate nil pointer check for better error
	if rcvr.IsPtr() && rcvr.Elem() == methodrcvr {
		// generating wrapper from *T to T.
		n := nod(OIF, nil, nil)
		n.Left = nod(OEQ, nthis, nodnil())
		call := nod(OCALL, syslook("panicwrap"), nil)
		n.Nbody.Set1(call)
		fn.Nbody.Append(n)
	}

	dot := adddot(nodSym(OXDOT, nthis, method.Sym))

	// generate call
	// It's not possible to use a tail call when dynamic linking on ppc64le. The
	// bad scenario is when a local call is made to the wrapper: the wrapper will
	// call the implementation, which might be in a different module and so set
	// the TOC to the appropriate value for that module. But if it returns
	// directly to the wrapper's caller, nothing will reset it to the correct
	// value for that function.
	if !instrumenting && rcvr.IsPtr() && methodrcvr.IsPtr() && method.Embedded != 0 && !isifacemethod(method.Type) && !(thearch.LinkArch.Name == "ppc64le" && Ctxt.Flag_dynlink) {
		// generate tail call: adjust pointer receiver and jump to embedded method.
		dot = dot.Left // skip final .M
		// TODO(mdempsky): Remove dependency on dotlist.
		if !dotlist[0].field.Type.IsPtr() {
			dot = nod(OADDR, dot, nil)
		}
		as := nod(OAS, nthis, convnop(dot, rcvr))
		fn.Nbody.Append(as)
		fn.Nbody.Append(nodSym(ORETJMP, nil, methodSym(methodrcvr, method.Sym)))
	} else {
		fn.Func.SetWrapper(true) // ignore frame for panic+recover matching
		call := nod(OCALL, dot, nil)
		call.List.Set(paramNnames(tfn.Type))
		call.SetIsDDD(tfn.Type.IsVariadic())
		if method.Type.NumResults() > 0 {
			n := nod(ORETURN, nil, nil)
			n.List.Set1(call)
			call = n
		}
		fn.Nbody.Append(call)
	}

	if false && Debug['r'] != 0 {
		dumplist("genwrapper body", fn.Nbody)
	}

	funcbody()
	if debug_dclstack != 0 {
		testdclstack()
	}

	fn = typecheck(fn, ctxStmt)

	Curfn = fn
	typecheckslice(fn.Nbody.Slice(), ctxStmt)

	// Inline calls within (*T).M wrappers. This is safe because we only
	// generate those wrappers within the same compilation unit as (T).M.
	// TODO(mdempsky): Investigate why we can't enable this more generally.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type && rcvr.Elem().Sym != nil {
		inlcalls(fn)
	}
	escAnalyze([]*Node{fn}, false)

	Curfn = nil
	funccompile(fn)
}

func paramNnames(ft *types.Type) []*Node {
	args := make([]*Node, ft.NumParams())
	for i, f := range ft.Params().FieldSlice() {
		args[i] = asNode(f.Nname)
	}
	return args
}

func hashmem(t *types.Type) *Node {
	sym := Runtimepkg.Lookup("memhash")

	n := newname(sym)
	n.SetClass(PFUNC)
	n.Sym.SetFunc(true)
	n.Type = functype(nil, []*Node{
		anonfield(types.NewPtr(t)),
		anonfield(types.Types[TUINTPTR]),
		anonfield(types.Types[TUINTPTR]),
	}, []*Node{
		anonfield(types.Types[TUINTPTR]),
	})
	return n
}

func ifacelookdot(s *types.Sym, t *types.Type, ignorecase bool) (m *types.Field, followptr bool) {
	if t == nil {
		return nil, false
	}

	path, ambig := dotpath(s, t, &m, ignorecase)
	if path == nil {
		if ambig {
			yyerror("%v.%v is ambiguous", t, s)
		}
		return nil, false
	}

	for _, d := range path {
		if d.field.Type.IsPtr() {
			followptr = true
			break
		}
	}

	if m.Type.Etype != TFUNC || m.Type.Recv() == nil {
		yyerror("%v.%v is a field, not a method", t, s)
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
			if false && Debug['r'] != 0 {
				yyerror("interface pointer mismatch")
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

func listtreecopy(l []*Node, pos src.XPos) []*Node {
	var out []*Node
	for _, n := range l {
		out = append(out, treecopy(n, pos))
	}
	return out
}

func liststmt(l []*Node) *Node {
	n := nod(OBLOCK, nil, nil)
	n.List.Set(l)
	if len(l) != 0 {
		n.Pos = l[0].Pos
	}
	return n
}

func (l Nodes) asblock() *Node {
	n := nod(OBLOCK, nil, nil)
	n.List = l
	if l.Len() != 0 {
		n.Pos = l.First().Pos
	}
	return n
}

func ngotype(n *Node) *types.Sym {
	if n.Type != nil {
		return typenamesym(n.Type)
	}
	return nil
}

// The result of addinit MUST be assigned back to n, e.g.
// 	n.Left = addinit(n.Left, init)
func addinit(n *Node, init []*Node) *Node {
	if len(init) == 0 {
		return n
	}
	if n.mayBeShared() {
		// Introduce OCONVNOP to hold init list.
		n = nod(OCONVNOP, n, nil)
		n.Type = n.Left.Type
		n.SetTypecheck(1)
	}

	n.Ninit.Prepend(init...)
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
		yyerror("import path contains NUL")
		return true
	}

	for _, ri := range reservedimports {
		if path == ri {
			yyerror("import path %q is reserved and cannot be used", path)
			return true
		}
	}

	for _, r := range path {
		if r == utf8.RuneError {
			yyerror("import path contains invalid UTF-8 sequence: %q", path)
			return true
		}

		if r < 0x20 || r == 0x7f {
			yyerror("import path contains control character: %q", path)
			return true
		}

		if r == '\\' {
			yyerror("import path contains backslash; use slash: %q", path)
			return true
		}

		if !allowSpace && unicode.IsSpace(r) {
			yyerror("import path contains space character: %q", path)
			return true
		}

		if strings.ContainsRune("!\"#$%&'()*,:;<=>?[]^`{|}", r) {
			yyerror("import path contains invalid character '%c': %q", r, path)
			return true
		}
	}

	return false
}

func checknil(x *Node, init *Nodes) {
	x = walkexpr(x, nil) // caller has not done this yet
	if x.Type.IsInterface() {
		x = nod(OITAB, x, nil)
		x = typecheck(x, ctxExpr)
	}

	n := nod(OCHECKNIL, x, nil)
	n.SetTypecheck(1)
	init.Append(n)
}

// Can this type be stored directly in an interface word?
// Yes, if the representation is a single pointer.
func isdirectiface(t *types.Type) bool {
	if t.Broke() {
		return false
	}

	switch t.Etype {
	case TPTR,
		TCHAN,
		TMAP,
		TFUNC,
		TUNSAFEPTR:
		return true

	case TARRAY:
		// Array of 1 direct iface type can be direct.
		return t.NumElem() == 1 && isdirectiface(t.Elem())

	case TSTRUCT:
		// Struct with 1 field of direct iface type can be direct.
		return t.NumFields() == 1 && isdirectiface(t.Field(0).Type)
	}

	return false
}

// itabType loads the _type field from a runtime.itab struct.
func itabType(itab *Node) *Node {
	typ := nodSym(ODOTPTR, itab, nil)
	typ.Type = types.NewPtr(types.Types[TUINT8])
	typ.SetTypecheck(1)
	typ.Xoffset = int64(Widthptr) // offset of _type in runtime.itab
	typ.SetBounded(true)          // guaranteed not to fault
	return typ
}

// ifaceData loads the data field from an interface.
// The concrete type must be known to have type t.
// It follows the pointer if !isdirectiface(t).
func ifaceData(n *Node, t *types.Type) *Node {
	ptr := nodSym(OIDATA, n, nil)
	if isdirectiface(t) {
		ptr.Type = t
		ptr.SetTypecheck(1)
		return ptr
	}
	ptr.Type = types.NewPtr(t)
	ptr.SetBounded(true)
	ptr.SetTypecheck(1)
	ind := nod(ODEREF, ptr, nil)
	ind.Type = t
	ind.SetTypecheck(1)
	return ind
}
