// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"cmd/internal/obj"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"os"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type Error struct {
	lineno int32
	msg    string
}

var errors []Error

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
	if len(errors) > 0 && errors[len(errors)-1].lineno == n.Lineno && errors[len(errors)-1].msg == old {
		errors[len(errors)-1].msg = fmt.Sprintf("%v: undefined: %v in %v\n", n.Line(), n.Left, n)
	}
}

func adderr(line int32, format string, args ...interface{}) {
	errors = append(errors, Error{
		lineno: line,
		msg:    fmt.Sprintf("%v: %s\n", linestr(line), fmt.Sprintf(format, args...)),
	})
}

// byLineno sorts errors by lineno.
type byLineno []Error

func (x byLineno) Len() int           { return len(x) }
func (x byLineno) Less(i, j int) bool { return x[i].lineno < x[j].lineno }
func (x byLineno) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }

// flusherrors sorts errors seen so far by line number, prints them to stdout,
// and empties the errors array.
func flusherrors() {
	Ctxt.Bso.Flush()
	if len(errors) == 0 {
		return
	}
	sort.Stable(byLineno(errors))
	for i := 0; i < len(errors); i++ {
		if i == 0 || errors[i].msg != errors[i-1].msg {
			fmt.Printf("%s", errors[i].msg)
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

func linestr(line int32) string {
	return Ctxt.Line(int(line))
}

// lasterror keeps track of the most recently issued error.
// It is used to avoid multiple error messages on the same
// line.
var lasterror struct {
	syntax int32  // line of last syntax error
	other  int32  // line of last non-syntax error
	msg    string // error message of last non-syntax error
}

func yyerrorl(line int32, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)

	if strings.HasPrefix(msg, "syntax error") {
		nsyntaxerrors++
		// only one syntax error per line, no matter what error
		if lasterror.syntax == line {
			return
		}
		lasterror.syntax = line
	} else {
		// only one of multiple equal non-syntax errors per line
		// (flusherrors shows only one of them, so we filter them
		// here as best as we can (they may not appear in order)
		// so that we don't count them here and exit early, and
		// then have nothing to show for.)
		if lasterror.other == line && lasterror.msg == msg {
			return
		}
		lasterror.other = line
		lasterror.msg = msg
	}

	adderr(line, "%s", msg)

	hcrash()
	nerrors++
	if nsavederrors+nerrors >= 10 && Debug['e'] == 0 {
		flusherrors()
		fmt.Printf("%v: too many errors\n", linestr(line))
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

func Warnl(line int32, fmt_ string, args ...interface{}) {
	adderr(line, fmt_, args...)
	if Debug['m'] != 0 {
		flusherrors()
	}
}

func Fatalf(fmt_ string, args ...interface{}) {
	flusherrors()

	fmt.Printf("%v: internal compiler error: ", linestr(lineno))
	fmt.Printf(fmt_, args...)
	fmt.Printf("\n")

	// If this is a released compiler version, ask for a bug report.
	if strings.HasPrefix(obj.Version, "release") {
		fmt.Printf("\n")
		fmt.Printf("Please file a bug report including a short program that triggers the error.\n")
		fmt.Printf("https://golang.org/issue/new\n")
	} else {
		// Not a release; dump a stack trace, too.
		fmt.Println()
		os.Stdout.Write(debug.Stack())
		fmt.Println()
	}

	hcrash()
	errorexit()
}

func linehistpragma(file string) {
	if Debug['i'] != 0 {
		fmt.Printf("pragma %s at line %v\n", file, linestr(lexlineno))
	}
	Ctxt.AddImport(file)
}

func linehistpush(file string) {
	if Debug['i'] != 0 {
		fmt.Printf("import %s at line %v\n", file, linestr(lexlineno))
	}
	Ctxt.LineHist.Push(int(lexlineno), file)
}

func linehistpop() {
	if Debug['i'] != 0 {
		fmt.Printf("end of import at line %v\n", linestr(lexlineno))
	}
	Ctxt.LineHist.Pop(int(lexlineno))
}

func linehistupdate(file string, off int) {
	if Debug['i'] != 0 {
		fmt.Printf("line %s at line %v\n", file, linestr(lexlineno))
	}
	Ctxt.LineHist.Update(int(lexlineno), file, off)
}

func setlineno(n *Node) int32 {
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
			lineno = n.Lineno
			if lineno == 0 {
				if Debug['K'] != 0 {
					Warn("setlineno: line 0")
				}
				lineno = lno
			}
		}
	}

	return lno
}

func lookup(name string) *Sym {
	return localpkg.Lookup(name)
}

func lookupf(format string, a ...interface{}) *Sym {
	return lookup(fmt.Sprintf(format, a...))
}

func lookupBytes(name []byte) *Sym {
	return localpkg.LookupBytes(name)
}

// lookupN looks up the symbol starting with prefix and ending with
// the decimal n. If prefix is too long, lookupN panics.
func lookupN(prefix string, n int) *Sym {
	var buf [20]byte // plenty long enough for all current users
	copy(buf[:], prefix)
	b := strconv.AppendInt(buf[:len(prefix)], int64(n), 10)
	return lookupBytes(b)
}

// autolabel generates a new Name node for use with
// an automatically generated label.
// prefix is a short mnemonic (e.g. ".s" for switch)
// to help with debugging.
// It should begin with "." to avoid conflicts with
// user labels.
func autolabel(prefix string) *Node {
	if prefix[0] != '.' {
		Fatalf("autolabel prefix must start with '.', have %q", prefix)
	}
	fn := Curfn
	if Curfn == nil {
		Fatalf("autolabel outside function")
	}
	n := fn.Func.Label
	fn.Func.Label++
	return newname(lookupN(prefix, int(n)))
}

var initSyms []*Sym

var nopkg = &Pkg{
	Syms: make(map[string]*Sym),
}

func (pkg *Pkg) Lookup(name string) *Sym {
	if pkg == nil {
		pkg = nopkg
	}
	if s := pkg.Syms[name]; s != nil {
		return s
	}

	s := &Sym{
		Name: name,
		Pkg:  pkg,
	}
	if name == "init" {
		initSyms = append(initSyms, s)
	}
	pkg.Syms[name] = s
	return s
}

func (pkg *Pkg) LookupBytes(name []byte) *Sym {
	if pkg == nil {
		pkg = nopkg
	}
	if s := pkg.Syms[string(name)]; s != nil {
		return s
	}
	str := internString(name)
	return pkg.Lookup(str)
}

func Pkglookup(name string, pkg *Pkg) *Sym {
	return pkg.Lookup(name)
}

func restrictlookup(name string, pkg *Pkg) *Sym {
	if !exportname(name) && pkg != localpkg {
		yyerror("cannot refer to unexported name %s.%s", pkg.Name, name)
	}
	return Pkglookup(name, pkg)
}

// find all the exported symbols in package opkg
// and make them available in the current package
func importdot(opkg *Pkg, pack *Node) {
	var s1 *Sym
	var pkgerror string

	n := 0
	for _, s := range opkg.Syms {
		if s.Def == nil {
			continue
		}
		if !exportname(s.Name) || strings.ContainsRune(s.Name, 0xb7) { // 0xb7 = center dot
			continue
		}
		s1 = lookup(s.Name)
		if s1.Def != nil {
			pkgerror = fmt.Sprintf("during import %q", opkg.Path)
			redeclare(s1, pkgerror)
			continue
		}

		s1.Def = s.Def
		s1.Block = s.Block
		if s1.Def.Name == nil {
			Dump("s1def", s1.Def)
			Fatalf("missing Name")
		}
		s1.Def.Name.Pack = pack
		s1.Origpkg = opkg
		n++
	}

	if n == 0 {
		// can't possibly be used - there were no symbols
		yyerrorl(pack.Lineno, "imported and not used: %q", opkg.Path)
	}
}

func nod(op Op, nleft *Node, nright *Node) *Node {
	n := new(Node)
	n.Op = op
	n.Left = nleft
	n.Right = nright
	n.Lineno = lineno
	n.Xoffset = BADWIDTH
	n.Orig = n
	switch op {
	case OCLOSURE, ODCLFUNC:
		n.Func = new(Func)
		n.Func.IsHiddenClosure = Curfn != nil
	case ONAME:
		n.Name = new(Name)
		n.Name.Param = new(Param)
	case OLABEL, OPACK:
		n.Name = new(Name)
	}
	if n.Name != nil {
		n.Name.Curfn = Curfn
	}
	return n
}

// nodSym makes a Node with Op op and with the Left field set to left
// and the Sym field set to sym. This is for ODOT and friends.
func nodSym(op Op, left *Node, sym *Sym) *Node {
	n := nod(op, left, nil)
	n.Sym = sym
	return n
}

func saveorignode(n *Node) {
	if n.Orig != nil {
		return
	}
	norig := nod(n.Op, nil, nil)
	*norig = *n
	n.Orig = norig
}

// methcmp sorts by symbol, then by package path for unexported symbols.
type methcmp []*Field

func (x methcmp) Len() int      { return len(x) }
func (x methcmp) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x methcmp) Less(i, j int) bool {
	a := x[i]
	b := x[j]
	if a.Sym == nil && b.Sym == nil {
		return false
	}
	if a.Sym == nil {
		return true
	}
	if b.Sym == nil {
		return false
	}
	if a.Sym.Name != b.Sym.Name {
		return a.Sym.Name < b.Sym.Name
	}
	if !exportname(a.Sym.Name) {
		if a.Sym.Pkg.Path != b.Sym.Pkg.Path {
			return a.Sym.Pkg.Path < b.Sym.Pkg.Path
		}
	}

	return false
}

func nodintconst(v int64) *Node {
	c := nod(OLITERAL, nil, nil)
	c.Addable = true
	c.SetVal(Val{new(Mpint)})
	c.Val().U.(*Mpint).SetInt64(v)
	c.Type = Types[TIDEAL]
	ullmancalc(c)
	return c
}

func nodfltconst(v *Mpflt) *Node {
	c := nod(OLITERAL, nil, nil)
	c.Addable = true
	c.SetVal(Val{newMpflt()})
	c.Val().U.(*Mpflt).Set(v)
	c.Type = Types[TIDEAL]
	ullmancalc(c)
	return c
}

func Nodconst(n *Node, t *Type, v int64) {
	*n = Node{}
	n.Op = OLITERAL
	n.Addable = true
	ullmancalc(n)
	n.SetVal(Val{new(Mpint)})
	n.Val().U.(*Mpint).SetInt64(v)
	n.Type = t

	if t.IsFloat() {
		Fatalf("nodconst: bad type %v", t)
	}
}

func nodnil() *Node {
	c := nodintconst(0)
	c.SetVal(Val{new(NilVal)})
	c.Type = Types[TNIL]
	return c
}

func nodbool(b bool) *Node {
	c := nodintconst(0)
	c.SetVal(Val{b})
	c.Type = idealbool
	return c
}

// treecopy recursively copies n, with the exception of
// ONAME, OLITERAL, OTYPE, and non-iota ONONAME leaves.
// Copies of iota ONONAME nodes are assigned the current
// value of iota_. If lineno != 0, it sets the line number
// of newly allocated nodes to lineno.
func treecopy(n *Node, lineno int32) *Node {
	if n == nil {
		return nil
	}

	switch n.Op {
	default:
		m := *n
		m.Orig = &m
		m.Left = treecopy(n.Left, lineno)
		m.Right = treecopy(n.Right, lineno)
		m.List.Set(listtreecopy(n.List.Slice(), lineno))
		if lineno != 0 {
			m.Lineno = lineno
		}
		if m.Name != nil && n.Op != ODCLFIELD {
			Dump("treecopy", n)
			Fatalf("treecopy Name")
		}
		return &m

	case ONONAME:
		if n.Sym == lookup("iota") {
			// Not sure yet whether this is the real iota,
			// but make a copy of the Node* just in case,
			// so that all the copies of this const definition
			// don't have the same iota value.
			m := *n
			if lineno != 0 {
				m.Lineno = lineno
			}
			m.SetIota(iota_)
			return &m
		}
		return n

	case OPACK:
		// OPACK nodes are never valid in const value declarations,
		// but allow them like any other declared symbol to avoid
		// crashing (golang.org/issue/11361).
		fallthrough

	case ONAME, OLITERAL, OTYPE:
		return n

	}
}

// isnil reports whether n represents the universal untyped zero value "nil".
func isnil(n *Node) bool {
	// Check n.Orig because constant propagation may produce typed nil constants,
	// which don't exist in the Go spec.
	return Isconst(n.Orig, CTNIL)
}

func isptrto(t *Type, et EType) bool {
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

func isblank(n *Node) bool {
	if n == nil {
		return false
	}
	return isblanksym(n.Sym)
}

func isblanksym(s *Sym) bool {
	return s != nil && s.Name == "_"
}

// methtype returns the underlying type, if any,
// that owns methods with receiver parameter t.
// The result is either a named type or an anonymous struct.
func methtype(t *Type) *Type {
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

func cplxsubtype(et EType) EType {
	switch et {
	case TCOMPLEX64:
		return TFLOAT32

	case TCOMPLEX128:
		return TFLOAT64
	}

	Fatalf("cplxsubtype: %v\n", et)
	return 0
}

// eqtype reports whether t1 and t2 are identical, following the spec rules.
//
// Any cyclic type must go through a named type, and if one is
// named, it is only identical to the other if they are the same
// pointer (t1 == t2), so there's no chance of chasing cycles
// ad infinitum, so no need for a depth counter.
func eqtype(t1, t2 *Type) bool {
	return eqtype1(t1, t2, true, nil)
}

// eqtypeIgnoreTags is like eqtype but it ignores struct tags for struct identity.
func eqtypeIgnoreTags(t1, t2 *Type) bool {
	return eqtype1(t1, t2, false, nil)
}

type typePair struct {
	t1 *Type
	t2 *Type
}

func eqtype1(t1, t2 *Type, cmpTags bool, assumedEqual map[typePair]struct{}) bool {
	if t1 == t2 {
		return true
	}
	if t1 == nil || t2 == nil || t1.Etype != t2.Etype || t1.Broke || t2.Broke {
		return false
	}
	if t1.Sym != nil || t2.Sym != nil {
		// Special case: we keep byte/uint8 and rune/int32
		// separate for error messages. Treat them as equal.
		switch t1.Etype {
		case TUINT8:
			return (t1 == Types[TUINT8] || t1 == bytetype) && (t2 == Types[TUINT8] || t2 == bytetype)
		case TINT32:
			return (t1 == Types[TINT32] || t1 == runetype) && (t2 == Types[TINT32] || t2 == runetype)
		default:
			return false
		}
	}

	if assumedEqual == nil {
		assumedEqual = make(map[typePair]struct{})
	} else if _, ok := assumedEqual[typePair{t1, t2}]; ok {
		return true
	}
	assumedEqual[typePair{t1, t2}] = struct{}{}

	switch t1.Etype {
	case TINTER, TSTRUCT:
		t1, i1 := iterFields(t1)
		t2, i2 := iterFields(t2)
		for ; t1 != nil && t2 != nil; t1, t2 = i1.Next(), i2.Next() {
			if t1.Sym != t2.Sym || t1.Embedded != t2.Embedded || !eqtype1(t1.Type, t2.Type, cmpTags, assumedEqual) || cmpTags && t1.Note != t2.Note {
				return false
			}
		}

		if t1 == nil && t2 == nil {
			return true
		}
		return false

	case TFUNC:
		// Check parameters and result parameters for type equality.
		// We intentionally ignore receiver parameters for type
		// equality, because they're never relevant.
		for _, f := range paramsResults {
			// Loop over fields in structs, ignoring argument names.
			ta, ia := iterFields(f(t1))
			tb, ib := iterFields(f(t2))
			for ; ta != nil && tb != nil; ta, tb = ia.Next(), ib.Next() {
				if ta.Isddd != tb.Isddd || !eqtype1(ta.Type, tb.Type, cmpTags, assumedEqual) {
					return false
				}
			}
			if ta != nil || tb != nil {
				return false
			}
		}
		return true

	case TARRAY:
		if t1.NumElem() != t2.NumElem() {
			return false
		}

	case TCHAN:
		if t1.ChanDir() != t2.ChanDir() {
			return false
		}

	case TMAP:
		if !eqtype1(t1.Key(), t2.Key(), cmpTags, assumedEqual) {
			return false
		}
		return eqtype1(t1.Val(), t2.Val(), cmpTags, assumedEqual)
	}

	return eqtype1(t1.Elem(), t2.Elem(), cmpTags, assumedEqual)
}

// Are t1 and t2 equal struct types when field names are ignored?
// For deciding whether the result struct from g can be copied
// directly when compiling f(g()).
func eqtypenoname(t1 *Type, t2 *Type) bool {
	if t1 == nil || t2 == nil || !t1.IsStruct() || !t2.IsStruct() {
		return false
	}

	f1, i1 := iterFields(t1)
	f2, i2 := iterFields(t2)
	for {
		if !eqtype(f1.Type, f2.Type) {
			return false
		}
		if f1 == nil {
			return true
		}
		f1 = i1.Next()
		f2 = i2.Next()
	}
}

// Is type src assignment compatible to type dst?
// If so, return op code to use in conversion.
// If not, return 0.
func assignop(src *Type, dst *Type, why *string) Op {
	if why != nil {
		*why = ""
	}

	// TODO(rsc,lvd): This behaves poorly in the presence of inlining.
	// https://golang.org/issue/2795
	if safemode && importpkg == nil && src != nil && src.Etype == TUNSAFEPTR {
		yyerror("cannot use unsafe.Pointer")
		errorexit()
	}

	if src == dst {
		return OCONVNOP
	}
	if src == nil || dst == nil || src.Etype == TFORW || dst.Etype == TFORW || src.Orig == nil || dst.Orig == nil {
		return 0
	}

	// 1. src type is identical to dst.
	if eqtype(src, dst) {
		return OCONVNOP
	}

	// 2. src and dst have identical underlying types
	// and either src or dst is not a named type or
	// both are empty interface types.
	// For assignable but different non-empty interface types,
	// we want to recompute the itab.
	if eqtype(src.Orig, dst.Orig) && (src.Sym == nil || dst.Sym == nil || src.IsEmptyInterface()) {
		return OCONVNOP
	}

	// 3. dst is an interface type and src implements dst.
	if dst.IsInterface() && src.Etype != TNIL {
		var missing, have *Field
		var ptr int
		if implements(src, dst, &missing, &have, &ptr) {
			return OCONVIFACE
		}

		// we'll have complained about this method anyway, suppress spurious messages.
		if have != nil && have.Sym == missing.Sym && (have.Type.Broke || missing.Type.Broke) {
			return OCONVIFACE
		}

		if why != nil {
			if isptrto(src, TINTER) {
				*why = fmt.Sprintf(":\n\t%v is pointer to interface, not interface", src)
			} else if have != nil && have.Sym == missing.Sym && have.Nointerface {
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
		var missing, have *Field
		var ptr int
		if why != nil && implements(dst, src, &missing, &have, &ptr) {
			*why = ": need type assertion"
		}
		return 0
	}

	// 4. src is a bidirectional channel value, dst is a channel type,
	// src and dst have identical element types, and
	// either src or dst is not a named type.
	if src.IsChan() && src.ChanDir() == Cboth && dst.IsChan() {
		if eqtype(src.Elem(), dst.Elem()) && (src.Sym == nil || dst.Sym == nil) {
			return OCONVNOP
		}
	}

	// 5. src is the predeclared identifier nil and dst is a nillable type.
	if src.Etype == TNIL {
		switch dst.Etype {
		case TPTR32,
			TPTR64,
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
func convertop(src *Type, dst *Type, why *string) Op {
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
	// (unless it's unsafe.Pointer). This is a runtime-specific
	// rule.
	if src.IsPtr() && dst.IsPtr() && dst.Elem().NotInHeap && !src.Elem().NotInHeap {
		if why != nil {
			*why = fmt.Sprintf(":\n\t%v is go:notinheap, but %v is not", dst.Elem(), src.Elem())
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
	if eqtypeIgnoreTags(src.Orig, dst.Orig) {
		return OCONVNOP
	}

	// 3. src and dst are unnamed pointer types and, ignoring struct tags,
	// their base types have identical underlying types.
	if src.IsPtr() && dst.IsPtr() && src.Sym == nil && dst.Sym == nil {
		if eqtypeIgnoreTags(src.Elem().Orig, dst.Elem().Orig) {
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
		if src.Elem().Etype == bytetype.Etype {
			return OARRAYBYTESTR
		}
		if src.Elem().Etype == runetype.Etype {
			return OARRAYRUNESTR
		}
	}

	// 7. src is a string and dst is []byte or []rune.
	// String to slice.
	if src.IsString() && dst.IsSlice() {
		if dst.Elem().Etype == bytetype.Etype {
			return OSTRARRAYBYTE
		}
		if dst.Elem().Etype == runetype.Etype {
			return OSTRARRAYRUNE
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

	return 0
}

func assignconv(n *Node, t *Type, context string) *Node {
	return assignconvfn(n, t, func() string { return context })
}

// Convert node n for assignment to type t.
func assignconvfn(n *Node, t *Type, context func() string) *Node {
	if n == nil || n.Type == nil || n.Type.Broke {
		return n
	}

	if t.Etype == TBLANK && n.Type.Etype == TNIL {
		yyerror("use of untyped nil")
	}

	old := n
	od := old.Diag
	old.Diag = true // silence errors about n; we'll issue one below
	n = defaultlit(n, t)
	old.Diag = od
	if t.Etype == TBLANK {
		return n
	}

	// Convert ideal bool from comparison to plain bool
	// if the next step is non-bool (like interface{}).
	if n.Type == idealbool && !t.IsBoolean() {
		if n.Op == ONAME || n.Op == OLITERAL {
			r := nod(OCONVNOP, n, nil)
			r.Type = Types[TBOOL]
			r.Typecheck = 1
			r.Implicit = true
			n = r
		}
	}

	if eqtype(n.Type, t) {
		return n
	}

	var why string
	op := assignop(n.Type, t, &why)
	if op == 0 {
		yyerror("cannot use %L as type %v in %s%s", n, t, context(), why)
		op = OCONV
	}

	r := nod(op, n, nil)
	r.Type = t
	r.Typecheck = 1
	r.Implicit = true
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
			n.List.Set([]*Node{low, high})
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
			n.List.Set([]*Node{low, high, max})
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

func syslook(name string) *Node {
	s := Pkglookup(name, Runtimepkg)
	if s == nil || s.Def == nil {
		Fatalf("syslook: can't find runtime.%s", name)
	}
	return s.Def
}

// typehash computes a hash value for type t to use in type switch
// statements.
func typehash(t *Type) uint32 {
	// t.tconv(FmtLeft | FmtUnsigned) already contains all the necessary logic
	// to generate a representation that completely describes the type, so using
	// it here avoids duplicating that code.
	// See the comments in exprSwitch.checkDupCases.
	p := t.tconv(FmtLeft | FmtUnsigned)

	// Using MD5 is overkill, but reduces accidental collisions.
	h := md5.Sum([]byte(p))
	return binary.LittleEndian.Uint32(h[:4])
}

// ptrto returns the Type *t.
// The returned struct must not be modified.
func ptrto(t *Type) *Type {
	if Tptr == 0 {
		Fatalf("ptrto: no tptr")
	}
	if t == nil {
		Fatalf("ptrto: nil ptr")
	}
	return typPtr(t)
}

func frame(context int) {
	if context != 0 {
		fmt.Printf("--- external frame ---\n")
		for _, n := range externdcl {
			printframenode(n)
		}
		return
	}

	if Curfn != nil {
		fmt.Printf("--- %v frame ---\n", Curfn.Func.Nname.Sym)
		for _, ln := range Curfn.Func.Dcl {
			printframenode(ln)
		}
	}
}

func printframenode(n *Node) {
	w := int64(-1)
	if n.Type != nil {
		w = n.Type.Width
	}
	switch n.Op {
	case ONAME:
		fmt.Printf("%v %v G%d %v width=%d\n", n.Op, n.Sym, n.Name.Vargen, n.Type, w)
	case OTYPE:
		fmt.Printf("%v %v width=%d\n", n.Op, n.Type, w)
	}
}

// calculate sethi/ullman number
// roughly how many registers needed to
// compile a node. used to compile the
// hardest side first to minimize registers.
func ullmancalc(n *Node) {
	if n == nil {
		return
	}

	var ul int
	var ur int
	if n.Ninit.Len() != 0 {
		ul = UINF
		goto out
	}

	switch n.Op {
	case OLITERAL, ONAME:
		ul = 1
		if n.Class == PAUTOHEAP {
			ul++
		}
		goto out

	case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER, OASWB:
		ul = UINF
		goto out

		// hard with instrumented code
	case OANDAND, OOROR:
		if instrumenting {
			ul = UINF
			goto out
		}
	case OINDEX, OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR, OSLICESTR,
		OIND, ODOTPTR, ODOTTYPE, ODIV, OMOD:
		// These ops might panic, make sure they are done
		// before we start marshaling args for a call. See issue 16760.
		ul = UINF
		goto out
	}

	ul = 1
	if n.Left != nil {
		ul = int(n.Left.Ullman)
	}
	ur = 1
	if n.Right != nil {
		ur = int(n.Right.Ullman)
	}
	if ul == ur {
		ul += 1
	}
	if ur > ul {
		ul = ur
	}

out:
	if ul > 200 {
		ul = 200 // clamp to uchar with room to grow
	}
	n.Ullman = uint8(ul)
}

func badtype(op Op, tl *Type, tr *Type) {
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
		r := nod(OXXX, nil, nil)
		*r = *n
		r.Left = l
		r = typecheck(r, Erv)
		r = walkexpr(r, init)
		return r

	case ODOTPTR, OIND:
		l := safeexpr(n.Left, init)
		if l == n.Left {
			return n
		}
		a := nod(OXXX, nil, nil)
		*a = *n
		a.Left = l
		a = walkexpr(a, init)
		return a

	case OINDEX, OINDEXMAP:
		l := safeexpr(n.Left, init)
		r := safeexpr(n.Right, init)
		if l == n.Left && r == n.Right {
			return n
		}
		a := nod(OXXX, nil, nil)
		*a = *n
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

func copyexpr(n *Node, t *Type, init *Nodes) *Node {
	l := temp(t)
	a := nod(OAS, l, n)
	a = typecheck(a, Etop)
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
	field *Field
}

// dotlist is used by adddot1 to record the path of embedded fields
// used to access a target field or method.
// Must be non-nil so that dotpath returns a non-nil slice even if d is zero.
var dotlist = make([]Dlist, 10)

// lookdot0 returns the number of fields or methods named s associated
// with Type t. If exactly one exists, it will be returned in *save
// (if save is not nil).
func lookdot0(s *Sym, t *Type, save **Field, ignorecase bool) int {
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
func adddot1(s *Sym, t *Type, d int, save **Field, ignorecase bool) (c int, more bool) {
	if t.Trecur != 0 {
		return
	}
	t.Trecur = 1

	var u *Type
	d--
	if d < 0 {
		// We've reached our target depth. If t has any fields/methods
		// named s, then we're done. Otherwise, we still need to check
		// below for embedded fields.
		c = lookdot0(s, t, save, ignorecase)
		if c != 0 {
			goto out
		}
	}

	u = t
	if u.IsPtr() {
		u = u.Elem()
	}
	if !u.IsStruct() && !u.IsInterface() {
		goto out
	}

	for _, f := range u.Fields().Slice() {
		if f.Embedded == 0 || f.Sym == nil {
			continue
		}
		if d < 0 {
			// Found an embedded field at target depth.
			more = true
			goto out
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

out:
	t.Trecur = 0
	return c, more
}

// dotpath computes the unique shortest explicit selector path to fully qualify
// a selection expression x.f, where x is of type t and f is the symbol s.
// If no such path exists, dotpath returns nil.
// If there are multiple shortest paths to the same depth, ambig is true.
func dotpath(s *Sym, t *Type, save **Field, ignorecase bool) (path []Dlist, ambig bool) {
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
	n.Left = typecheck(n.Left, Etype|Erv)
	if n.Left.Diag {
		n.Diag = true
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
			n.Left.Implicit = true
		}
	case ambig:
		yyerror("ambiguous selector %v", n)
		n.Left = nil
	}

	return n
}

// code to help generate trampoline
// functions for methods on embedded
// subtypes.
// these are approx the same as
// the corresponding adddot routines
// except that they expect to be called
// with unique tasks and they return
// the actual methods.
type Symlink struct {
	field     *Field
	followptr bool
}

var slist []Symlink

func expand0(t *Type, followptr bool) {
	u := t
	if u.IsPtr() {
		followptr = true
		u = u.Elem()
	}

	if u.IsInterface() {
		for _, f := range u.Fields().Slice() {
			if f.Sym.Flags&SymUniq != 0 {
				continue
			}
			f.Sym.Flags |= SymUniq
			slist = append(slist, Symlink{field: f, followptr: followptr})
		}

		return
	}

	u = methtype(t)
	if u != nil {
		for _, f := range u.Methods().Slice() {
			if f.Sym.Flags&SymUniq != 0 {
				continue
			}
			f.Sym.Flags |= SymUniq
			slist = append(slist, Symlink{field: f, followptr: followptr})
		}
	}
}

func expand1(t *Type, top, followptr bool) {
	if t.Trecur != 0 {
		return
	}
	t.Trecur = 1

	if !top {
		expand0(t, followptr)
	}

	u := t
	if u.IsPtr() {
		followptr = true
		u = u.Elem()
	}

	if !u.IsStruct() && !u.IsInterface() {
		goto out
	}

	for _, f := range u.Fields().Slice() {
		if f.Embedded == 0 {
			continue
		}
		if f.Sym == nil {
			continue
		}
		expand1(f.Type, false, followptr)
	}

out:
	t.Trecur = 0
}

func expandmeth(t *Type) {
	if t == nil || t.AllMethods().Len() != 0 {
		return
	}

	// mark top-level method symbols
	// so that expand1 doesn't consider them.
	for _, f := range t.Methods().Slice() {
		f.Sym.Flags |= SymUniq
	}

	// generate all reachable methods
	slist = slist[:0]
	expand1(t, true, false)

	// check each method to be uniquely reachable
	var ms []*Field
	for i, sl := range slist {
		slist[i].field = nil
		sl.field.Sym.Flags &^= SymUniq

		var f *Field
		if path, _ := dotpath(sl.field.Sym, t, &f, false); path == nil {
			continue
		}

		// dotpath may have dug out arbitrary fields, we only want methods.
		if f.Type.Etype != TFUNC || f.Type.Recv() == nil {
			continue
		}

		// add it to the base type method list
		f = f.Copy()
		f.Embedded = 1 // needs a trampoline
		if sl.followptr {
			f.Embedded = 2
		}
		ms = append(ms, f)
	}

	for _, f := range t.Methods().Slice() {
		f.Sym.Flags &^= SymUniq
	}

	ms = append(ms, t.Methods().Slice()...)
	t.AllMethods().Set(ms)
}

// Given funarg struct list, return list of ODCLFIELD Node fn args.
func structargs(tl *Type, mustname bool) []*Node {
	var args []*Node
	gen := 0
	for _, t := range tl.Fields().Slice() {
		var n *Node
		if mustname && (t.Sym == nil || t.Sym.Name == "_") {
			// invent a name so that we can refer to it in the trampoline
			buf := fmt.Sprintf(".anon%d", gen)
			gen++
			n = newname(lookup(buf))
		} else if t.Sym != nil {
			n = newname(t.Sym)
		}
		a := nod(ODCLFIELD, n, typenod(t.Type))
		a.Isddd = t.Isddd
		if n != nil {
			n.Isddd = t.Isddd
		}
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

var genwrapper_linehistdone int = 0

func genwrapper(rcvr *Type, method *Field, newnam *Sym, iface int) {
	if false && Debug['r'] != 0 {
		fmt.Printf("genwrapper rcvrtype=%v method=%v newnam=%v\n", rcvr, method, newnam)
	}

	lexlineno++
	lineno = lexlineno
	if genwrapper_linehistdone == 0 {
		// All the wrappers can share the same linehist entry.
		linehistpush("<autogenerated>")

		genwrapper_linehistdone = 1
	}

	dclcontext = PEXTERN
	markdcl()

	this := nod(ODCLFIELD, newname(lookup(".this")), typenod(rcvr))
	this.Left.Name.Param.Ntype = this.Right
	in := structargs(method.Type.Params(), true)
	out := structargs(method.Type.Results(), false)

	t := nod(OTFUNC, nil, nil)
	l := []*Node{this}
	if iface != 0 && rcvr.Width < Types[Tptr].Width {
		// Building method for interface table and receiver
		// is smaller than the single pointer-sized word
		// that the interface call will pass in.
		// Add a dummy padding argument after the
		// receiver to make up the difference.
		tpad := typArray(Types[TUINT8], Types[Tptr].Width-rcvr.Width)
		pad := nod(ODCLFIELD, newname(lookup(".pad")), typenod(tpad))
		l = append(l, pad)
	}

	t.List.Set(append(l, in...))
	t.Rlist.Set(out)

	fn := nod(ODCLFUNC, nil, nil)
	fn.Func.Nname = newname(newnam)
	fn.Func.Nname.Name.Defn = fn
	fn.Func.Nname.Name.Param.Ntype = t
	declare(fn.Func.Nname, PFUNC)
	funchdr(fn)

	// arg list
	var args []*Node

	isddd := false
	for _, n := range in {
		args = append(args, n.Left)
		isddd = n.Left.Isddd
	}

	methodrcvr := method.Type.Recv().Type

	// generate nil pointer check for better error
	if rcvr.IsPtr() && rcvr.Elem() == methodrcvr {
		// generating wrapper from *T to T.
		n := nod(OIF, nil, nil)

		n.Left = nod(OEQ, this.Left, nodnil())

		// these strings are already in the reflect tables,
		// so no space cost to use them here.
		var l []*Node

		var v Val
		v.U = rcvr.Elem().Sym.Pkg.Name // package name
		l = append(l, nodlit(v))
		v.U = rcvr.Elem().Sym.Name // type name
		l = append(l, nodlit(v))
		v.U = method.Sym.Name
		l = append(l, nodlit(v)) // method name
		call := nod(OCALL, syslook("panicwrap"), nil)
		call.List.Set(l)
		n.Nbody.Set1(call)
		fn.Nbody.Append(n)
	}

	dot := adddot(nodSym(OXDOT, this.Left, method.Sym))

	// generate call
	// It's not possible to use a tail call when dynamic linking on ppc64le. The
	// bad scenario is when a local call is made to the wrapper: the wrapper will
	// call the implementation, which might be in a different module and so set
	// the TOC to the appropriate value for that module. But if it returns
	// directly to the wrapper's caller, nothing will reset it to the correct
	// value for that function.
	if !instrumenting && rcvr.IsPtr() && methodrcvr.IsPtr() && method.Embedded != 0 && !isifacemethod(method.Type) && !(Thearch.LinkArch.Name == "ppc64le" && Ctxt.Flag_dynlink) {
		// generate tail call: adjust pointer receiver and jump to embedded method.
		dot = dot.Left // skip final .M
		// TODO(mdempsky): Remove dependency on dotlist.
		if !dotlist[0].field.Type.IsPtr() {
			dot = nod(OADDR, dot, nil)
		}
		as := nod(OAS, this.Left, nod(OCONVNOP, dot, nil))
		as.Right.Type = rcvr
		fn.Nbody.Append(as)
		n := nod(ORETJMP, nil, nil)
		n.Left = newname(methodsym(method.Sym, methodrcvr, 0))
		fn.Nbody.Append(n)
		// When tail-calling, we can't use a frame pointer.
		fn.Func.NoFramePointer = true
	} else {
		fn.Func.Wrapper = true // ignore frame for panic+recover matching
		call := nod(OCALL, dot, nil)
		call.List.Set(args)
		call.Isddd = isddd
		if method.Type.Results().NumFields() > 0 {
			n := nod(ORETURN, nil, nil)
			n.List.Set1(call)
			call = n
		}

		fn.Nbody.Append(call)
	}

	if false && Debug['r'] != 0 {
		dumplist("genwrapper body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn
	popdcl()
	if debug_dclstack != 0 {
		testdclstack()
	}

	// wrappers where T is anonymous (struct or interface) can be duplicated.
	if rcvr.IsStruct() || rcvr.IsInterface() || rcvr.IsPtr() && rcvr.Elem().IsStruct() {
		fn.Func.Dupok = true
	}
	fn = typecheck(fn, Etop)
	typecheckslice(fn.Nbody.Slice(), Etop)

	inlcalls(fn)
	escAnalyze([]*Node{fn}, false)

	Curfn = nil
	funccompile(fn)
}

func hashmem(t *Type) *Node {
	sym := Pkglookup("memhash", Runtimepkg)

	n := newname(sym)
	n.Class = PFUNC
	tfn := nod(OTFUNC, nil, nil)
	tfn.List.Append(nod(ODCLFIELD, nil, typenod(ptrto(t))))
	tfn.List.Append(nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn.List.Append(nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn.Rlist.Append(nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn = typecheck(tfn, Etype)
	n.Type = tfn.Type
	return n
}

func ifacelookdot(s *Sym, t *Type, followptr *bool, ignorecase bool) *Field {
	*followptr = false

	if t == nil {
		return nil
	}

	var m *Field
	path, ambig := dotpath(s, t, &m, ignorecase)
	if path == nil {
		if ambig {
			yyerror("%v.%v is ambiguous", t, s)
		}
		return nil
	}

	for _, d := range path {
		if d.field.Type.IsPtr() {
			*followptr = true
			break
		}
	}

	if m.Type.Etype != TFUNC || m.Type.Recv() == nil {
		yyerror("%v.%v is a field, not a method", t, s)
		return nil
	}

	return m
}

func implements(t, iface *Type, m, samename **Field, ptr *int) bool {
	t0 := t
	if t == nil {
		return false
	}

	// if this is too slow,
	// could sort these first
	// and then do one loop.

	if t.IsInterface() {
		for _, im := range iface.Fields().Slice() {
			for _, tm := range t.Fields().Slice() {
				if tm.Sym == im.Sym {
					if eqtype(tm.Type, im.Type) {
						goto found
					}
					*m = im
					*samename = tm
					*ptr = 0
					return false
				}
			}

			*m = im
			*samename = nil
			*ptr = 0
			return false
		found:
		}

		return true
	}

	t = methtype(t)
	if t != nil {
		expandmeth(t)
	}
	for _, im := range iface.Fields().Slice() {
		if im.Broke {
			continue
		}
		var followptr bool
		tm := ifacelookdot(im.Sym, t, &followptr, false)
		if tm == nil || tm.Nointerface || !eqtype(tm.Type, im.Type) {
			if tm == nil {
				tm = ifacelookdot(im.Sym, t, &followptr, true)
			}
			*m = im
			*samename = tm
			*ptr = 0
			return false
		}

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

	return true
}

// even simpler simtype; get rid of ptr, bool.
// assuming that the front end has rejected
// all the invalid conversions (like ptr -> bool)
func Simsimtype(t *Type) EType {
	if t == nil {
		return 0
	}

	et := simtype[t.Etype]
	switch et {
	case TPTR32:
		et = TUINT32

	case TPTR64:
		et = TUINT64

	case TBOOL:
		et = TUINT8
	}

	return et
}

func listtreecopy(l []*Node, lineno int32) []*Node {
	var out []*Node
	for _, n := range l {
		out = append(out, treecopy(n, lineno))
	}
	return out
}

func liststmt(l []*Node) *Node {
	n := nod(OBLOCK, nil, nil)
	n.List.Set(l)
	if len(l) != 0 {
		n.Lineno = l[0].Lineno
	}
	return n
}

// return power of 2 of the constant
// operand. -1 if it is not a power of 2.
// 1000+ if it is a -(power of 2)
func powtwo(n *Node) int {
	if n == nil || n.Op != OLITERAL || n.Type == nil {
		return -1
	}
	if !n.Type.IsInteger() {
		return -1
	}

	v := uint64(n.Int64())
	b := uint64(1)
	for i := 0; i < 64; i++ {
		if b == v {
			return i
		}
		b = b << 1
	}

	if !n.Type.IsSigned() {
		return -1
	}

	v = -v
	b = 1
	for i := 0; i < 64; i++ {
		if b == v {
			return i + 1000
		}
		b = b << 1
	}

	return -1
}

func ngotype(n *Node) *Sym {
	if n.Type != nil {
		return typenamesym(n.Type)
	}
	return nil
}

// Convert raw string to the prefix that will be used in the symbol
// table. All control characters, space, '%' and '"', as well as
// non-7-bit clean bytes turn into %xx. The period needs escaping
// only in the last segment of the path, and it makes for happier
// users if we escape that as little as possible.
//
// If you edit this, edit ../../debug/goobj/read.go:/importPathToPrefix too.
func pathtoprefix(s string) string {
	slash := strings.LastIndex(s, "/")
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c <= ' ' || i >= slash && c == '.' || c == '%' || c == '"' || c >= 0x7F {
			var buf bytes.Buffer
			for i := 0; i < len(s); i++ {
				c := s[i]
				if c <= ' ' || i >= slash && c == '.' || c == '%' || c == '"' || c >= 0x7F {
					fmt.Fprintf(&buf, "%%%02x", c)
					continue
				}
				buf.WriteByte(c)
			}
			return buf.String()
		}
	}
	return s
}

var pkgMap = make(map[string]*Pkg)
var pkgs []*Pkg

func mkpkg(path string) *Pkg {
	if p := pkgMap[path]; p != nil {
		return p
	}

	p := new(Pkg)
	p.Path = path
	p.Prefix = pathtoprefix(path)
	p.Syms = make(map[string]*Sym)
	pkgMap[path] = p
	pkgs = append(pkgs, p)
	return p
}

// The result of addinit MUST be assigned back to n, e.g.
// 	n.Left = addinit(n.Left, init)
func addinit(n *Node, init []*Node) *Node {
	if len(init) == 0 {
		return n
	}

	switch n.Op {
	// There may be multiple refs to this node;
	// introduce OCONVNOP to hold init list.
	case ONAME, OLITERAL:
		n = nod(OCONVNOP, n, nil)
		n.Type = n.Left.Type
		n.Typecheck = 1
	}

	n.Ninit.Prepend(init...)
	n.Ullman = UINF
	return n
}

var reservedimports = []string{
	"go",
	"type",
}

func isbadimport(path string) bool {
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

		if unicode.IsSpace(r) {
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
		x = typecheck(x, Erv)
	}

	n := nod(OCHECKNIL, x, nil)
	n.Typecheck = 1
	init.Append(n)
}

// Can this type be stored directly in an interface word?
// Yes, if the representation is a single pointer.
func isdirectiface(t *Type) bool {
	switch t.Etype {
	case TPTR32,
		TPTR64,
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
	typ.Type = ptrto(Types[TUINT8])
	typ.Typecheck = 1
	typ.Xoffset = int64(Widthptr) // offset of _type in runtime.itab
	typ.Bounded = true            // guaranteed not to fault
	return typ
}

// ifaceData loads the data field from an interface.
// The concrete type must be known to have type t.
// It follows the pointer if !isdirectiface(t).
func ifaceData(n *Node, t *Type) *Node {
	ptr := nodSym(OIDATA, n, nil)
	if isdirectiface(t) {
		ptr.Type = t
		ptr.Typecheck = 1
		return ptr
	}
	ptr.Type = ptrto(t)
	ptr.Bounded = true
	ptr.Typecheck = 1
	ind := nod(OIND, ptr, nil)
	ind.Type = t
	ind.Typecheck = 1
	return ind
}

// iet returns 'T' if t is a concrete type,
// 'I' if t is an interface type, and 'E' if t is an empty interface type.
// It is used to build calls to the conv* and assert* runtime routines.
func (t *Type) iet() byte {
	if t.IsEmptyInterface() {
		return 'E'
	}
	if t.IsInterface() {
		return 'I'
	}
	return 'T'
}
