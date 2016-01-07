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
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

type Error struct {
	lineno int
	seq    int
	msg    string
}

var errors []Error

func errorexit() {
	Flusherrors()
	if outfile != "" {
		os.Remove(outfile)
	}
	os.Exit(2)
}

func parserline() int {
	return int(lineno)
}

func adderrorname(n *Node) {
	if n.Op != ODOT {
		return
	}
	old := fmt.Sprintf("%v: undefined: %v\n", n.Line(), n.Left)
	if len(errors) > 0 && int32(errors[len(errors)-1].lineno) == n.Lineno && errors[len(errors)-1].msg == old {
		errors[len(errors)-1].msg = fmt.Sprintf("%v: undefined: %v in %v\n", n.Line(), n.Left, n)
	}
}

func adderr(line int, format string, args ...interface{}) {
	errors = append(errors, Error{
		seq:    len(errors),
		lineno: line,
		msg:    fmt.Sprintf("%v: %s\n", Ctxt.Line(line), fmt.Sprintf(format, args...)),
	})
}

// errcmp sorts errors by line, then seq, then message.
type errcmp []Error

func (x errcmp) Len() int      { return len(x) }
func (x errcmp) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x errcmp) Less(i, j int) bool {
	a := &x[i]
	b := &x[j]
	if a.lineno != b.lineno {
		return a.lineno < b.lineno
	}
	if a.seq != b.seq {
		return a.seq < b.seq
	}
	return a.msg < b.msg
}

func Flusherrors() {
	bstdout.Flush()
	if len(errors) == 0 {
		return
	}
	sort.Sort(errcmp(errors))
	for i := 0; i < len(errors); i++ {
		if i == 0 || errors[i].msg != errors[i-1].msg {
			fmt.Printf("%s", errors[i].msg)
		}
	}
	errors = errors[:0]
}

func hcrash() {
	if Debug['h'] != 0 {
		Flusherrors()
		if outfile != "" {
			os.Remove(outfile)
		}
		var x *int
		*x = 0
	}
}

func yyerrorl(line int, format string, args ...interface{}) {
	adderr(line, format, args...)

	hcrash()
	nerrors++
	if nsavederrors+nerrors >= 10 && Debug['e'] == 0 {
		Flusherrors()
		fmt.Printf("%v: too many errors\n", Ctxt.Line(line))
		errorexit()
	}
}

var yyerror_lastsyntax int

func Yyerror(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	if strings.HasPrefix(msg, "syntax error") {
		nsyntaxerrors++

		// An unexpected EOF caused a syntax error. Use the previous
		// line number since getc generated a fake newline character.
		if curio.eofnl {
			lexlineno = prevlineno
		}

		// only one syntax error per line
		if int32(yyerror_lastsyntax) == lexlineno {
			return
		}
		yyerror_lastsyntax = int(lexlineno)

		// plain "syntax error" gets "near foo" added
		if msg == "syntax error" {
			yyerrorl(int(lexlineno), "syntax error near %s", lexbuf.String())
			return
		}

		yyerrorl(int(lexlineno), "%s", msg)
		return
	}

	adderr(parserline(), "%s", msg)

	hcrash()
	nerrors++
	if nsavederrors+nerrors >= 10 && Debug['e'] == 0 {
		Flusherrors()
		fmt.Printf("%v: too many errors\n", Ctxt.Line(parserline()))
		errorexit()
	}
}

func Warn(fmt_ string, args ...interface{}) {
	adderr(parserline(), fmt_, args...)

	hcrash()
}

func Warnl(line int, fmt_ string, args ...interface{}) {
	adderr(line, fmt_, args...)
	if Debug['m'] != 0 {
		Flusherrors()
	}
}

func Fatalf(fmt_ string, args ...interface{}) {
	Flusherrors()

	fmt.Printf("%v: internal compiler error: ", Ctxt.Line(int(lineno)))
	fmt.Printf(fmt_, args...)
	fmt.Printf("\n")

	// If this is a released compiler version, ask for a bug report.
	if strings.HasPrefix(obj.Getgoversion(), "release") {
		fmt.Printf("\n")
		fmt.Printf("Please file a bug report including a short program that triggers the error.\n")
		fmt.Printf("https://golang.org/issue/new\n")
	}

	hcrash()
	errorexit()
}

func linehistpragma(file string) {
	if Debug['i'] != 0 {
		fmt.Printf("pragma %s at line %v\n", file, Ctxt.Line(int(lexlineno)))
	}
	Ctxt.AddImport(file)
}

func linehistpush(file string) {
	if Debug['i'] != 0 {
		fmt.Printf("import %s at line %v\n", file, Ctxt.Line(int(lexlineno)))
	}
	Ctxt.LineHist.Push(int(lexlineno), file)
}

func linehistpop() {
	if Debug['i'] != 0 {
		fmt.Printf("end of import at line %v\n", Ctxt.Line(int(lexlineno)))
	}
	Ctxt.LineHist.Pop(int(lexlineno))
}

func linehistupdate(file string, off int) {
	if Debug['i'] != 0 {
		fmt.Printf("line %s at line %v\n", file, Ctxt.Line(int(lexlineno)))
	}
	Ctxt.LineHist.Update(int(lexlineno), file, off)
}

func setlineno(n *Node) int32 {
	lno := lineno
	if n != nil {
		switch n.Op {
		case ONAME, OTYPE, OPACK:
			break

		case OLITERAL:
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

func Lookup(name string) *Sym {
	return localpkg.Lookup(name)
}

func Lookupf(format string, a ...interface{}) *Sym {
	return Lookup(fmt.Sprintf(format, a...))
}

func LookupBytes(name []byte) *Sym {
	return localpkg.LookupBytes(name)
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
		Name:    name,
		Pkg:     pkg,
		Lexical: LNAME,
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
		Yyerror("cannot refer to unexported name %s.%s", pkg.Name, name)
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
		s1 = Lookup(s.Name)
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
		yyerrorl(int(pack.Lineno), "imported and not used: %q", opkg.Path)
	}
}

func Nod(op Op, nleft *Node, nright *Node) *Node {
	n := new(Node)
	n.Op = op
	n.Left = nleft
	n.Right = nright
	n.Lineno = int32(parserline())
	n.Xoffset = BADWIDTH
	n.Orig = n
	switch op {
	case OCLOSURE, ODCLFUNC:
		n.Func = new(Func)
		n.Func.FCurfn = Curfn
	case ONAME:
		n.Name = new(Name)
		n.Name.Param = new(Param)
	case OLABEL, OPACK:
		n.Name = new(Name)
	case ODCLFIELD:
		if nleft != nil {
			n.Name = nleft.Name
		} else {
			n.Name = new(Name)
			n.Name.Param = new(Param)
		}
	}
	if n.Name != nil {
		n.Name.Curfn = Curfn
	}
	return n
}

func saveorignode(n *Node) {
	if n.Orig != nil {
		return
	}
	norig := Nod(n.Op, nil, nil)
	*norig = *n
	n.Orig = norig
}

// ispaddedfield reports whether the given field
// is followed by padding. For the case where t is
// the last field, total gives the size of the enclosing struct.
func ispaddedfield(t *Type, total int64) bool {
	if t.Etype != TFIELD {
		Fatalf("ispaddedfield called non-field %v", t)
	}
	if t.Down == nil {
		return t.Width+t.Type.Width != total
	}
	return t.Width+t.Type.Width != t.Down.Width
}

func algtype1(t *Type, bad **Type) int {
	if bad != nil {
		*bad = nil
	}
	if t.Broke {
		return AMEM
	}
	if t.Noalg {
		return ANOEQ
	}

	switch t.Etype {
	// will be defined later.
	case TANY, TFORW:
		*bad = t

		return -1

	case TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TINT,
		TUINT,
		TUINTPTR,
		TBOOL,
		TPTR32,
		TPTR64,
		TCHAN,
		TUNSAFEPTR:
		return AMEM

	case TFUNC, TMAP:
		if bad != nil {
			*bad = t
		}
		return ANOEQ

	case TFLOAT32:
		return AFLOAT32

	case TFLOAT64:
		return AFLOAT64

	case TCOMPLEX64:
		return ACPLX64

	case TCOMPLEX128:
		return ACPLX128

	case TSTRING:
		return ASTRING

	case TINTER:
		if isnilinter(t) {
			return ANILINTER
		}
		return AINTER

	case TARRAY:
		if Isslice(t) {
			if bad != nil {
				*bad = t
			}
			return ANOEQ
		}

		a := algtype1(t.Type, bad)
		if a == ANOEQ || a == AMEM {
			if a == ANOEQ && bad != nil {
				*bad = t
			}
			return a
		}

		return -1 // needs special compare

	case TSTRUCT:
		if t.Type != nil && t.Type.Down == nil && !isblanksym(t.Type.Sym) {
			// One-field struct is same as that one field alone.
			return algtype1(t.Type.Type, bad)
		}

		ret := AMEM
		var a int
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			// All fields must be comparable.
			a = algtype1(t1.Type, bad)

			if a == ANOEQ {
				return ANOEQ
			}

			// Blank fields, padded fields, fields with non-memory
			// equality need special compare.
			if a != AMEM || isblanksym(t1.Sym) || ispaddedfield(t1, t.Width) {
				ret = -1
				continue
			}
		}

		return ret
	}

	Fatalf("algtype1: unexpected type %v", t)
	return 0
}

func algtype(t *Type) int {
	a := algtype1(t, nil)
	if a == AMEM || a == ANOEQ {
		if Isslice(t) {
			return ASLICE
		}
		switch t.Width {
		case 0:
			return a + AMEM0 - AMEM

		case 1:
			return a + AMEM8 - AMEM

		case 2:
			return a + AMEM16 - AMEM

		case 4:
			return a + AMEM32 - AMEM

		case 8:
			return a + AMEM64 - AMEM

		case 16:
			return a + AMEM128 - AMEM
		}
	}

	return a
}

func maptype(key *Type, val *Type) *Type {
	if key != nil {
		var bad *Type
		atype := algtype1(key, &bad)
		var mtype EType
		if bad == nil {
			mtype = key.Etype
		} else {
			mtype = bad.Etype
		}
		switch mtype {
		default:
			if atype == ANOEQ {
				Yyerror("invalid map key type %v", key)
			}

			// will be resolved later.
		case TANY:
			break

			// map[key] used during definition of key.
		// postpone check until key is fully defined.
		// if there are multiple uses of map[key]
		// before key is fully defined, the error
		// will only be printed for the first one.
		// good enough.
		case TFORW:
			if key.Maplineno == 0 {
				key.Maplineno = lineno
			}
		}
	}

	t := typ(TMAP)
	t.Down = key
	t.Type = val
	return t
}

func typ(et EType) *Type {
	t := new(Type)
	t.Etype = et
	t.Width = BADWIDTH
	t.Lineno = int(lineno)
	t.Orig = t
	return t
}

// methcmp sorts by symbol, then by package path for unexported symbols.
type methcmp []*Type

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

func sortinter(t *Type) *Type {
	if t.Type == nil || t.Type.Down == nil {
		return t
	}

	var a []*Type
	for f := t.Type; f != nil; f = f.Down {
		a = append(a, f)
	}
	sort.Sort(methcmp(a))

	n := len(a) // n > 0 due to initial conditions.
	for i := 0; i < n-1; i++ {
		a[i].Down = a[i+1]
	}
	a[n-1].Down = nil

	t.Type = a[0]
	return t
}

func Nodintconst(v int64) *Node {
	c := Nod(OLITERAL, nil, nil)
	c.Addable = true
	c.SetVal(Val{new(Mpint)})
	Mpmovecfix(c.Val().U.(*Mpint), v)
	c.Type = Types[TIDEAL]
	ullmancalc(c)
	return c
}

func nodfltconst(v *Mpflt) *Node {
	c := Nod(OLITERAL, nil, nil)
	c.Addable = true
	c.SetVal(Val{newMpflt()})
	mpmovefltflt(c.Val().U.(*Mpflt), v)
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
	Mpmovecfix(n.Val().U.(*Mpint), v)
	n.Type = t

	if Isfloat[t.Etype] {
		Fatalf("nodconst: bad type %v", t)
	}
}

func nodnil() *Node {
	c := Nodintconst(0)
	c.SetVal(Val{new(NilVal)})
	c.Type = Types[TNIL]
	return c
}

func Nodbool(b bool) *Node {
	c := Nodintconst(0)
	c.SetVal(Val{b})
	c.Type = idealbool
	return c
}

func aindex(b *Node, t *Type) *Type {
	bound := int64(-1) // open bound
	typecheck(&b, Erv)
	if b != nil {
		switch consttype(b) {
		default:
			Yyerror("array bound must be an integer expression")

		case CTINT, CTRUNE:
			bound = Mpgetfix(b.Val().U.(*Mpint))
			if bound < 0 {
				Yyerror("array bound must be non negative")
			}
		}
	}

	// fixed array
	r := typ(TARRAY)

	r.Type = t
	r.Bound = bound
	return r
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

	var m *Node
	switch n.Op {
	default:
		m = Nod(OXXX, nil, nil)
		*m = *n
		m.Orig = m
		m.Left = treecopy(n.Left, lineno)
		m.Right = treecopy(n.Right, lineno)
		m.List = listtreecopy(n.List, lineno)
		if lineno != 0 {
			m.Lineno = lineno
		}
		if m.Name != nil && n.Op != ODCLFIELD {
			Dump("treecopy", n)
			Fatalf("treecopy Name")
		}

	case ONONAME:
		if n.Sym == Lookup("iota") {
			// Not sure yet whether this is the real iota,
			// but make a copy of the Node* just in case,
			// so that all the copies of this const definition
			// don't have the same iota value.
			m = Nod(OXXX, nil, nil)
			*m = *n
			if lineno != 0 {
				m.Lineno = lineno
			}
			m.Name = new(Name)
			*m.Name = *n.Name
			m.Name.Iota = iota_
			break
		}
		fallthrough

	case ONAME, OLITERAL, OTYPE:
		m = n
	}

	return m
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
	if !Isptr[t.Etype] {
		return false
	}
	t = t.Type
	if t == nil {
		return false
	}
	if t.Etype != et {
		return false
	}
	return true
}

func Istype(t *Type, et EType) bool {
	return t != nil && t.Etype == et
}

func Isfixedarray(t *Type) bool {
	return t != nil && t.Etype == TARRAY && t.Bound >= 0
}

func Isslice(t *Type) bool {
	return t != nil && t.Etype == TARRAY && t.Bound < 0
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

func Isinter(t *Type) bool {
	return t != nil && t.Etype == TINTER
}

func isnilinter(t *Type) bool {
	if !Isinter(t) {
		return false
	}
	if t.Type != nil {
		return false
	}
	return true
}

func isideal(t *Type) bool {
	if t == nil {
		return false
	}
	if t == idealstring || t == idealbool {
		return true
	}
	switch t.Etype {
	case TNIL, TIDEAL:
		return true
	}

	return false
}

// given receiver of type t (t == r or t == *r)
// return type to hang methods off (r).
func methtype(t *Type, mustname int) *Type {
	if t == nil {
		return nil
	}

	// strip away pointer if it's there
	if Isptr[t.Etype] {
		if t.Sym != nil {
			return nil
		}
		t = t.Type
		if t == nil {
			return nil
		}
	}

	// need a type name
	if t.Sym == nil && (mustname != 0 || t.Etype != TSTRUCT) {
		return nil
	}

	// check types
	if !issimple[t.Etype] {
		switch t.Etype {
		default:
			return nil

		case TSTRUCT,
			TARRAY,
			TMAP,
			TCHAN,
			TSTRING,
			TFUNC:
			break
		}
	}

	return t
}

func cplxsubtype(et EType) EType {
	switch et {
	case TCOMPLEX64:
		return TFLOAT32

	case TCOMPLEX128:
		return TFLOAT64
	}

	Fatalf("cplxsubtype: %v\n", Econv(et))
	return 0
}

func eqnote(a, b *string) bool {
	return a == b || a != nil && b != nil && *a == *b
}

type TypePairList struct {
	t1   *Type
	t2   *Type
	next *TypePairList
}

func onlist(l *TypePairList, t1 *Type, t2 *Type) bool {
	for ; l != nil; l = l.next {
		if (l.t1 == t1 && l.t2 == t2) || (l.t1 == t2 && l.t2 == t1) {
			return true
		}
	}
	return false
}

// Return 1 if t1 and t2 are identical, following the spec rules.
//
// Any cyclic type must go through a named type, and if one is
// named, it is only identical to the other if they are the same
// pointer (t1 == t2), so there's no chance of chasing cycles
// ad infinitum, so no need for a depth counter.
func Eqtype(t1 *Type, t2 *Type) bool {
	return eqtype1(t1, t2, nil)
}

func eqtype1(t1 *Type, t2 *Type, assumed_equal *TypePairList) bool {
	if t1 == t2 {
		return true
	}
	if t1 == nil || t2 == nil || t1.Etype != t2.Etype {
		return false
	}
	if t1.Sym != nil || t2.Sym != nil {
		// Special case: we keep byte and uint8 separate
		// for error messages.  Treat them as equal.
		switch t1.Etype {
		case TUINT8:
			if (t1 == Types[TUINT8] || t1 == bytetype) && (t2 == Types[TUINT8] || t2 == bytetype) {
				return true
			}

		case TINT, TINT32:
			if (t1 == Types[runetype.Etype] || t1 == runetype) && (t2 == Types[runetype.Etype] || t2 == runetype) {
				return true
			}
		}

		return false
	}

	if onlist(assumed_equal, t1, t2) {
		return true
	}
	var l TypePairList
	l.next = assumed_equal
	l.t1 = t1
	l.t2 = t2

	switch t1.Etype {
	case TINTER, TSTRUCT:
		t1 = t1.Type
		t2 = t2.Type
		for ; t1 != nil && t2 != nil; t1, t2 = t1.Down, t2.Down {
			if t1.Etype != TFIELD || t2.Etype != TFIELD {
				Fatalf("struct/interface missing field: %v %v", t1, t2)
			}
			if t1.Sym != t2.Sym || t1.Embedded != t2.Embedded || !eqtype1(t1.Type, t2.Type, &l) || !eqnote(t1.Note, t2.Note) {
				return false
			}
		}

		if t1 == nil && t2 == nil {
			return true
		}
		return false

		// Loop over structs: receiver, in, out.
	case TFUNC:
		t1 = t1.Type
		t2 = t2.Type
		for ; t1 != nil && t2 != nil; t1, t2 = t1.Down, t2.Down {
			if t1.Etype != TSTRUCT || t2.Etype != TSTRUCT {
				Fatalf("func missing struct: %v %v", t1, t2)
			}

			// Loop over fields in structs, ignoring argument names.
			ta := t1.Type
			tb := t2.Type
			for ; ta != nil && tb != nil; ta, tb = ta.Down, tb.Down {
				if ta.Etype != TFIELD || tb.Etype != TFIELD {
					Fatalf("func struct missing field: %v %v", ta, tb)
				}
				if ta.Isddd != tb.Isddd || !eqtype1(ta.Type, tb.Type, &l) {
					return false
				}
			}

			if ta != nil || tb != nil {
				return false
			}
		}

		if t1 == nil && t2 == nil {
			return true
		}
		return false

	case TARRAY:
		if t1.Bound != t2.Bound {
			return false
		}

	case TCHAN:
		if t1.Chan != t2.Chan {
			return false
		}
	}

	if eqtype1(t1.Down, t2.Down, &l) && eqtype1(t1.Type, t2.Type, &l) {
		return true
	}
	return false
}

// Are t1 and t2 equal struct types when field names are ignored?
// For deciding whether the result struct from g can be copied
// directly when compiling f(g()).
func eqtypenoname(t1 *Type, t2 *Type) bool {
	if t1 == nil || t2 == nil || t1.Etype != TSTRUCT || t2.Etype != TSTRUCT {
		return false
	}

	t1 = t1.Type
	t2 = t2.Type
	for {
		if !Eqtype(t1, t2) {
			return false
		}
		if t1 == nil {
			return true
		}
		t1 = t1.Down
		t2 = t2.Down
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
	if safemode != 0 && importpkg == nil && src != nil && src.Etype == TUNSAFEPTR {
		Yyerror("cannot use unsafe.Pointer")
		errorexit()
	}

	if src == dst {
		return OCONVNOP
	}
	if src == nil || dst == nil || src.Etype == TFORW || dst.Etype == TFORW || src.Orig == nil || dst.Orig == nil {
		return 0
	}

	// 1. src type is identical to dst.
	if Eqtype(src, dst) {
		return OCONVNOP
	}

	// 2. src and dst have identical underlying types
	// and either src or dst is not a named type or
	// both are empty interface types.
	// For assignable but different non-empty interface types,
	// we want to recompute the itab.
	if Eqtype(src.Orig, dst.Orig) && (src.Sym == nil || dst.Sym == nil || isnilinter(src)) {
		return OCONVNOP
	}

	// 3. dst is an interface type and src implements dst.
	if dst.Etype == TINTER && src.Etype != TNIL {
		var missing *Type
		var ptr int
		var have *Type
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
				*why = fmt.Sprintf(":\n\t%v does not implement %v (wrong type for %v method)\n"+"\t\thave %v%v\n\t\twant %v%v", src, dst, missing.Sym, have.Sym, Tconv(have.Type, obj.FmtShort|obj.FmtByte), missing.Sym, Tconv(missing.Type, obj.FmtShort|obj.FmtByte))
			} else if ptr != 0 {
				*why = fmt.Sprintf(":\n\t%v does not implement %v (%v method has pointer receiver)", src, dst, missing.Sym)
			} else if have != nil {
				*why = fmt.Sprintf(":\n\t%v does not implement %v (missing %v method)\n"+"\t\thave %v%v\n\t\twant %v%v", src, dst, missing.Sym, have.Sym, Tconv(have.Type, obj.FmtShort|obj.FmtByte), missing.Sym, Tconv(missing.Type, obj.FmtShort|obj.FmtByte))
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

	if src.Etype == TINTER && dst.Etype != TBLANK {
		var have *Type
		var ptr int
		var missing *Type
		if why != nil && implements(dst, src, &missing, &have, &ptr) {
			*why = ": need type assertion"
		}
		return 0
	}

	// 4. src is a bidirectional channel value, dst is a channel type,
	// src and dst have identical element types, and
	// either src or dst is not a named type.
	if src.Etype == TCHAN && src.Chan == Cboth && dst.Etype == TCHAN {
		if Eqtype(src.Type, dst.Type) && (src.Sym == nil || dst.Sym == nil) {
			return OCONVNOP
		}
	}

	// 5. src is the predeclared identifier nil and dst is a nillable type.
	if src.Etype == TNIL {
		switch dst.Etype {
		case TARRAY:
			if dst.Bound != -100 { // not slice
				break
			}
			fallthrough

		case TPTR32,
			TPTR64,
			TFUNC,
			TMAP,
			TCHAN,
			TINTER:
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

	// 1. src can be assigned to dst.
	op := assignop(src, dst, why)
	if op != 0 {
		return op
	}

	// The rules for interfaces are no different in conversions
	// than assignments.  If interfaces are involved, stop now
	// with the good message from assignop.
	// Otherwise clear the error.
	if src.Etype == TINTER || dst.Etype == TINTER {
		return 0
	}
	if why != nil {
		*why = ""
	}

	// 2. src and dst have identical underlying types.
	if Eqtype(src.Orig, dst.Orig) {
		return OCONVNOP
	}

	// 3. src and dst are unnamed pointer types
	// and their base types have identical underlying types.
	if Isptr[src.Etype] && Isptr[dst.Etype] && src.Sym == nil && dst.Sym == nil {
		if Eqtype(src.Type.Orig, dst.Type.Orig) {
			return OCONVNOP
		}
	}

	// 4. src and dst are both integer or floating point types.
	if (Isint[src.Etype] || Isfloat[src.Etype]) && (Isint[dst.Etype] || Isfloat[dst.Etype]) {
		if Simtype[src.Etype] == Simtype[dst.Etype] {
			return OCONVNOP
		}
		return OCONV
	}

	// 5. src and dst are both complex types.
	if Iscomplex[src.Etype] && Iscomplex[dst.Etype] {
		if Simtype[src.Etype] == Simtype[dst.Etype] {
			return OCONVNOP
		}
		return OCONV
	}

	// 6. src is an integer or has type []byte or []rune
	// and dst is a string type.
	if Isint[src.Etype] && dst.Etype == TSTRING {
		return ORUNESTR
	}

	if Isslice(src) && dst.Etype == TSTRING {
		if src.Type.Etype == bytetype.Etype {
			return OARRAYBYTESTR
		}
		if src.Type.Etype == runetype.Etype {
			return OARRAYRUNESTR
		}
	}

	// 7. src is a string and dst is []byte or []rune.
	// String to slice.
	if src.Etype == TSTRING && Isslice(dst) {
		if dst.Type.Etype == bytetype.Etype {
			return OSTRARRAYBYTE
		}
		if dst.Type.Etype == runetype.Etype {
			return OSTRARRAYRUNE
		}
	}

	// 8. src is a pointer or uintptr and dst is unsafe.Pointer.
	if (Isptr[src.Etype] || src.Etype == TUINTPTR) && dst.Etype == TUNSAFEPTR {
		return OCONVNOP
	}

	// 9. src is unsafe.Pointer and dst is a pointer or uintptr.
	if src.Etype == TUNSAFEPTR && (Isptr[dst.Etype] || dst.Etype == TUINTPTR) {
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
		Yyerror("use of untyped nil")
	}

	old := n
	old.Diag++ // silence errors about n; we'll issue one below
	defaultlit(&n, t)
	old.Diag--
	if t.Etype == TBLANK {
		return n
	}

	// Convert ideal bool from comparison to plain bool
	// if the next step is non-bool (like interface{}).
	if n.Type == idealbool && t.Etype != TBOOL {
		if n.Op == ONAME || n.Op == OLITERAL {
			r := Nod(OCONVNOP, n, nil)
			r.Type = Types[TBOOL]
			r.Typecheck = 1
			r.Implicit = true
			n = r
		}
	}

	if Eqtype(n.Type, t) {
		return n
	}

	var why string
	op := assignop(n.Type, t, &why)
	if op == 0 {
		Yyerror("cannot use %v as type %v in %s%s", Nconv(n, obj.FmtLong), t, context(), why)
		op = OCONV
	}

	r := Nod(op, n, nil)
	r.Type = t
	r.Typecheck = 1
	r.Implicit = true
	r.Orig = n.Orig
	return r
}

// substArgTypes substitutes the given list of types for
// successive occurrences of the "any" placeholder in the
// type syntax expression n.Type.
func substArgTypes(n *Node, types ...*Type) {
	for _, t := range types {
		dowidth(t)
	}
	substAny(&n.Type, &types)
	if len(types) > 0 {
		Fatalf("substArgTypes: too many argument types")
	}
}

// substAny walks *tp, replacing instances of "any" with successive
// elements removed from types.
func substAny(tp **Type, types *[]*Type) {
	for {
		t := *tp
		if t == nil {
			return
		}
		if t.Etype == TANY && t.Copyany {
			if len(*types) == 0 {
				Fatalf("substArgTypes: not enough argument types")
			}
			*tp = (*types)[0]
			*types = (*types)[1:]
		}

		switch t.Etype {
		case TPTR32, TPTR64, TCHAN, TARRAY:
			tp = &t.Type
			continue

		case TMAP:
			substAny(&t.Down, types)
			tp = &t.Type
			continue

		case TFUNC:
			substAny(&t.Type, types)
			substAny(&t.Type.Down.Down, types)
			substAny(&t.Type.Down, types)

		case TSTRUCT:
			for t = t.Type; t != nil; t = t.Down {
				substAny(&t.Type, types)
			}
		}
		return
	}
}

// Is this a 64-bit type?
func Is64(t *Type) bool {
	if t == nil {
		return false
	}
	switch Simtype[t.Etype] {
	case TINT64, TUINT64, TPTR64:
		return true
	}

	return false
}

// Is a conversion between t1 and t2 a no-op?
func Noconv(t1 *Type, t2 *Type) bool {
	e1 := Simtype[t1.Etype]
	e2 := Simtype[t2.Etype]

	switch e1 {
	case TINT8, TUINT8:
		return e2 == TINT8 || e2 == TUINT8

	case TINT16, TUINT16:
		return e2 == TINT16 || e2 == TUINT16

	case TINT32, TUINT32, TPTR32:
		return e2 == TINT32 || e2 == TUINT32 || e2 == TPTR32

	case TINT64, TUINT64, TPTR64:
		return e2 == TINT64 || e2 == TUINT64 || e2 == TPTR64

	case TFLOAT32:
		return e2 == TFLOAT32

	case TFLOAT64:
		return e2 == TFLOAT64
	}

	return false
}

func shallow(t *Type) *Type {
	if t == nil {
		return nil
	}
	nt := typ(0)
	*nt = *t
	if t.Orig == t {
		nt.Orig = nt
	}
	return nt
}

func deep(t *Type) *Type {
	if t == nil {
		return nil
	}

	var nt *Type
	switch t.Etype {
	default:
		nt = t // share from here down

	case TANY:
		nt = shallow(t)
		nt.Copyany = true

	case TPTR32, TPTR64, TCHAN, TARRAY:
		nt = shallow(t)
		nt.Type = deep(t.Type)

	case TMAP:
		nt = shallow(t)
		nt.Down = deep(t.Down)
		nt.Type = deep(t.Type)

	case TFUNC:
		nt = shallow(t)
		nt.Type = deep(t.Type)
		nt.Type.Down = deep(t.Type.Down)
		nt.Type.Down.Down = deep(t.Type.Down.Down)

	case TSTRUCT:
		nt = shallow(t)
		nt.Type = shallow(t.Type)
		xt := nt.Type

		for t = t.Type; t != nil; t = t.Down {
			xt.Type = deep(t.Type)
			xt.Down = shallow(t.Down)
			xt = xt.Down
		}
	}

	return nt
}

func syslook(name string, copy int) *Node {
	s := Pkglookup(name, Runtimepkg)
	if s == nil || s.Def == nil {
		Fatalf("syslook: can't find runtime.%s", name)
	}

	if copy == 0 {
		return s.Def
	}

	n := Nod(0, nil, nil)
	*n = *s.Def
	n.Type = deep(s.Def.Type)

	return n
}

// compute a hash value for type t.
// if t is a method type, ignore the receiver
// so that the hash can be used in interface checks.
// %T already contains
// all the necessary logic to generate a representation
// of the type that completely describes it.
// using smprint here avoids duplicating that code.
// using md5 here is overkill, but i got tired of
// accidental collisions making the runtime think
// two types are equal when they really aren't.
func typehash(t *Type) uint32 {
	var p string

	if t.Thistuple != 0 {
		// hide method receiver from Tpretty
		t.Thistuple = 0

		p = Tconv(t, obj.FmtLeft|obj.FmtUnsigned)
		t.Thistuple = 1
	} else {
		p = Tconv(t, obj.FmtLeft|obj.FmtUnsigned)
	}

	//print("typehash: %s\n", p);
	h := md5.Sum([]byte(p))
	return binary.LittleEndian.Uint32(h[:4])
}

var initPtrtoDone bool

var (
	ptrToUint8  *Type
	ptrToAny    *Type
	ptrToString *Type
	ptrToBool   *Type
	ptrToInt32  *Type
)

func initPtrto() {
	ptrToUint8 = ptrto1(Types[TUINT8])
	ptrToAny = ptrto1(Types[TANY])
	ptrToString = ptrto1(Types[TSTRING])
	ptrToBool = ptrto1(Types[TBOOL])
	ptrToInt32 = ptrto1(Types[TINT32])
}

func ptrto1(t *Type) *Type {
	t1 := typ(Tptr)
	t1.Type = t
	t1.Width = int64(Widthptr)
	t1.Align = uint8(Widthptr)
	return t1
}

// Ptrto returns the Type *t.
// The returned struct must not be modified.
func Ptrto(t *Type) *Type {
	if Tptr == 0 {
		Fatalf("ptrto: no tptr")
	}
	// Reduce allocations by pre-creating common cases.
	if !initPtrtoDone {
		initPtrto()
		initPtrtoDone = true
	}
	switch t {
	case Types[TUINT8]:
		return ptrToUint8
	case Types[TINT32]:
		return ptrToInt32
	case Types[TANY]:
		return ptrToAny
	case Types[TSTRING]:
		return ptrToString
	case Types[TBOOL]:
		return ptrToBool
	}
	return ptrto1(t)
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
		for l := Curfn.Func.Dcl; l != nil; l = l.Next {
			printframenode(l.N)
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
		fmt.Printf("%v %v G%d %v width=%d\n", Oconv(int(n.Op), 0), n.Sym, n.Name.Vargen, n.Type, w)
	case OTYPE:
		fmt.Printf("%v %v width=%d\n", Oconv(int(n.Op), 0), n.Type, w)
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
	if n.Ninit != nil {
		ul = UINF
		goto out
	}

	switch n.Op {
	case OREGISTER, OLITERAL, ONAME:
		ul = 1
		if n.Class == PPARAMREF || (n.Class&PHEAP != 0) {
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
	if tl != nil && tr != nil && Isptr[tl.Etype] && Isptr[tr.Etype] {
		if tl.Type.Etype == TSTRUCT && tr.Type.Etype == TINTER {
			fmt_ += "\n\t(*struct vs *interface)"
		} else if tl.Type.Etype == TINTER && tr.Type.Etype == TSTRUCT {
			fmt_ += "\n\t(*interface vs *struct)"
		}
	}

	s := fmt_
	Yyerror("illegal types for operand: %v%s", Oconv(int(op), 0), s)
}

// iterator to walk a structure declaration
func Structfirst(s *Iter, nn **Type) *Type {
	var t *Type

	n := *nn
	if n == nil {
		goto bad
	}

	switch n.Etype {
	default:
		goto bad

	case TSTRUCT, TINTER, TFUNC:
		break
	}

	t = n.Type
	if t == nil {
		return nil
	}

	if t.Etype != TFIELD {
		Fatalf("structfirst: not field %v", t)
	}

	s.T = t
	return t

bad:
	Fatalf("structfirst: not struct %v", n)

	return nil
}

func structnext(s *Iter) *Type {
	n := s.T
	t := n.Down
	if t == nil {
		return nil
	}

	if t.Etype != TFIELD {
		Fatalf("structnext: not struct %v", n)

		return nil
	}

	s.T = t
	return t
}

// iterator to this and inargs in a function
func funcfirst(s *Iter, t *Type) *Type {
	var fp *Type

	if t == nil {
		goto bad
	}

	if t.Etype != TFUNC {
		goto bad
	}

	s.Tfunc = t
	s.Done = 0
	fp = Structfirst(s, getthis(t))
	if fp == nil {
		s.Done = 1
		fp = Structfirst(s, getinarg(t))
	}

	return fp

bad:
	Fatalf("funcfirst: not func %v", t)
	return nil
}

func funcnext(s *Iter) *Type {
	fp := structnext(s)
	if fp == nil && s.Done == 0 {
		s.Done = 1
		fp = Structfirst(s, getinarg(s.Tfunc))
	}

	return fp
}

func getthis(t *Type) **Type {
	if t.Etype != TFUNC {
		Fatalf("getthis: not a func %v", t)
	}
	return &t.Type
}

func Getoutarg(t *Type) **Type {
	if t.Etype != TFUNC {
		Fatalf("getoutarg: not a func %v", t)
	}
	return &t.Type.Down
}

func getinarg(t *Type) **Type {
	if t.Etype != TFUNC {
		Fatalf("getinarg: not a func %v", t)
	}
	return &t.Type.Down.Down
}

func getthisx(t *Type) *Type {
	return *getthis(t)
}

func getoutargx(t *Type) *Type {
	return *Getoutarg(t)
}

func getinargx(t *Type) *Type {
	return *getinarg(t)
}

// Brcom returns !(op).
// For example, Brcom(==) is !=.
func Brcom(op Op) Op {
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
	Fatalf("brcom: no com for %v\n", Oconv(int(op), 0))
	return op
}

// Brrev returns reverse(op).
// For example, Brrev(<) is >.
func Brrev(op Op) Op {
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
	Fatalf("brrev: no rev for %v\n", Oconv(int(op), 0))
	return op
}

// return side effect-free n, appending side effects to init.
// result is assignable if n is.
func safeexpr(n *Node, init **NodeList) *Node {
	if n == nil {
		return nil
	}

	if n.Ninit != nil {
		walkstmtlist(n.Ninit)
		*init = concat(*init, n.Ninit)
		n.Ninit = nil
	}

	switch n.Op {
	case ONAME, OLITERAL:
		return n

	case ODOT, OLEN, OCAP:
		l := safeexpr(n.Left, init)
		if l == n.Left {
			return n
		}
		r := Nod(OXXX, nil, nil)
		*r = *n
		r.Left = l
		typecheck(&r, Erv)
		walkexpr(&r, init)
		return r

	case ODOTPTR, OIND:
		l := safeexpr(n.Left, init)
		if l == n.Left {
			return n
		}
		a := Nod(OXXX, nil, nil)
		*a = *n
		a.Left = l
		walkexpr(&a, init)
		return a

	case OINDEX, OINDEXMAP:
		l := safeexpr(n.Left, init)
		r := safeexpr(n.Right, init)
		if l == n.Left && r == n.Right {
			return n
		}
		a := Nod(OXXX, nil, nil)
		*a = *n
		a.Left = l
		a.Right = r
		walkexpr(&a, init)
		return a
	}

	// make a copy; must not be used as an lvalue
	if islvalue(n) {
		Fatalf("missing lvalue case in safeexpr: %v", n)
	}
	return cheapexpr(n, init)
}

func copyexpr(n *Node, t *Type, init **NodeList) *Node {
	l := temp(t)
	a := Nod(OAS, l, n)
	typecheck(&a, Etop)
	walkexpr(&a, init)
	*init = list(*init, a)
	return l
}

// return side-effect free and cheap n, appending side effects to init.
// result may not be assignable.
func cheapexpr(n *Node, init **NodeList) *Node {
	switch n.Op {
	case ONAME, OLITERAL:
		return n
	}

	return copyexpr(n, n.Type, init)
}

func Setmaxarg(t *Type, extra int32) {
	dowidth(t)
	w := t.Argwid
	if w >= Thearch.MAXWIDTH {
		Fatalf("bad argwid %v", t)
	}
	w += int64(extra)
	if w >= Thearch.MAXWIDTH {
		Fatalf("bad argwid %d + %v", extra, t)
	}
	if w > Maxarg {
		Maxarg = w
	}
}

// unicode-aware case-insensitive strcmp

// code to resolve elided DOTs
// in embedded types

// search depth 0 --
// return count of fields+methods
// found with a given name
func lookdot0(s *Sym, t *Type, save **Type, ignorecase int) int {
	u := t
	if Isptr[u.Etype] {
		u = u.Type
	}

	c := 0
	if u.Etype == TSTRUCT || u.Etype == TINTER {
		for f := u.Type; f != nil; f = f.Down {
			if f.Sym == s || (ignorecase != 0 && f.Type.Etype == TFUNC && f.Type.Thistuple > 0 && strings.EqualFold(f.Sym.Name, s.Name)) {
				if save != nil {
					*save = f
				}
				c++
			}
		}
	}

	u = methtype(t, 0)
	if u != nil {
		for f := u.Method; f != nil; f = f.Down {
			if f.Embedded == 0 && (f.Sym == s || (ignorecase != 0 && strings.EqualFold(f.Sym.Name, s.Name))) {
				if save != nil {
					*save = f
				}
				c++
			}
		}
	}

	return c
}

// search depth d for field/method s --
// return count of fields+methods
// found at search depth.
// answer is in dotlist array and
// count of number of ways is returned.
func adddot1(s *Sym, t *Type, d int, save **Type, ignorecase int) int {
	if t.Trecur != 0 {
		return 0
	}
	t.Trecur = 1

	var c int
	var u *Type
	var a int
	if d == 0 {
		c = lookdot0(s, t, save, ignorecase)
		goto out
	}

	c = 0
	u = t
	if Isptr[u.Etype] {
		u = u.Type
	}
	if u.Etype != TSTRUCT && u.Etype != TINTER {
		goto out
	}

	d--
	for f := u.Type; f != nil; f = f.Down {
		if f.Embedded == 0 {
			continue
		}
		if f.Sym == nil {
			continue
		}
		a = adddot1(s, f.Type, d, save, ignorecase)
		if a != 0 && c == 0 {
			dotlist[d].field = f
		}
		c += a
	}

out:
	t.Trecur = 0
	return c
}

// in T.field
// find missing fields that
// will give shortest unique addressing.
// modify the tree with missing type names.
func adddot(n *Node) *Node {
	typecheck(&n.Left, Etype|Erv)
	n.Diag |= n.Left.Diag
	t := n.Left.Type
	if t == nil {
		return n
	}

	if n.Left.Op == OTYPE {
		return n
	}

	if n.Right.Op != ONAME {
		return n
	}
	s := n.Right.Sym
	if s == nil {
		return n
	}

	var c int
	for d := 0; d < len(dotlist); d++ {
		c = adddot1(s, t, d, nil, 0)
		if c > 0 {
			if c > 1 {
				Yyerror("ambiguous selector %v", n)
				n.Left = nil
				return n
			}

			// rebuild elided dots
			for c := d - 1; c >= 0; c-- {
				n.Left = Nod(ODOT, n.Left, newname(dotlist[c].field.Sym))
				n.Left.Implicit = true
			}

			return n
		}
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
	field     *Type
	link      *Symlink
	good      bool
	followptr bool
}

var slist *Symlink

func expand0(t *Type, followptr bool) {
	u := t
	if Isptr[u.Etype] {
		followptr = true
		u = u.Type
	}

	if u.Etype == TINTER {
		var sl *Symlink
		for f := u.Type; f != nil; f = f.Down {
			if f.Sym.Flags&SymUniq != 0 {
				continue
			}
			f.Sym.Flags |= SymUniq
			sl = new(Symlink)
			sl.field = f
			sl.link = slist
			sl.followptr = followptr
			slist = sl
		}

		return
	}

	u = methtype(t, 0)
	if u != nil {
		var sl *Symlink
		for f := u.Method; f != nil; f = f.Down {
			if f.Sym.Flags&SymUniq != 0 {
				continue
			}
			f.Sym.Flags |= SymUniq
			sl = new(Symlink)
			sl.field = f
			sl.link = slist
			sl.followptr = followptr
			slist = sl
		}
	}
}

func expand1(t *Type, d int, followptr bool) {
	if t.Trecur != 0 {
		return
	}
	if d == 0 {
		return
	}
	t.Trecur = 1

	if d != len(dotlist)-1 {
		expand0(t, followptr)
	}

	u := t
	if Isptr[u.Etype] {
		followptr = true
		u = u.Type
	}

	if u.Etype != TSTRUCT && u.Etype != TINTER {
		goto out
	}

	for f := u.Type; f != nil; f = f.Down {
		if f.Embedded == 0 {
			continue
		}
		if f.Sym == nil {
			continue
		}
		expand1(f.Type, d-1, followptr)
	}

out:
	t.Trecur = 0
}

func expandmeth(t *Type) {
	if t == nil || t.Xmethod != nil {
		return
	}

	// mark top-level method symbols
	// so that expand1 doesn't consider them.
	var f *Type
	for f = t.Method; f != nil; f = f.Down {
		f.Sym.Flags |= SymUniq
	}

	// generate all reachable methods
	slist = nil

	expand1(t, len(dotlist)-1, false)

	// check each method to be uniquely reachable
	var c int
	var d int
	for sl := slist; sl != nil; sl = sl.link {
		sl.field.Sym.Flags &^= SymUniq
		for d = 0; d < len(dotlist); d++ {
			c = adddot1(sl.field.Sym, t, d, &f, 0)
			if c == 0 {
				continue
			}
			if c == 1 {
				// addot1 may have dug out arbitrary fields, we only want methods.
				if f.Type.Etype == TFUNC && f.Type.Thistuple > 0 {
					sl.good = true
					sl.field = f
				}
			}

			break
		}
	}

	for f = t.Method; f != nil; f = f.Down {
		f.Sym.Flags &^= SymUniq
	}

	t.Xmethod = t.Method
	for sl := slist; sl != nil; sl = sl.link {
		if sl.good {
			// add it to the base type method list
			f = typ(TFIELD)

			*f = *sl.field
			f.Embedded = 1 // needs a trampoline
			if sl.followptr {
				f.Embedded = 2
			}
			f.Down = t.Xmethod
			t.Xmethod = f
		}
	}
}

// Given funarg struct list, return list of ODCLFIELD Node fn args.
func structargs(tl **Type, mustname int) *NodeList {
	var savet Iter
	var a *Node
	var n *Node
	var buf string

	var args *NodeList
	gen := 0
	for t := Structfirst(&savet, tl); t != nil; t = structnext(&savet) {
		n = nil
		if mustname != 0 && (t.Sym == nil || t.Sym.Name == "_") {
			// invent a name so that we can refer to it in the trampoline
			buf = fmt.Sprintf(".anon%d", gen)
			gen++

			n = newname(Lookup(buf))
		} else if t.Sym != nil {
			n = newname(t.Sym)
		}
		a = Nod(ODCLFIELD, n, typenod(t.Type))
		a.Isddd = t.Isddd
		if n != nil {
			n.Isddd = t.Isddd
		}
		args = list(args, a)
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

func genwrapper(rcvr *Type, method *Type, newnam *Sym, iface int) {
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

	this := Nod(ODCLFIELD, newname(Lookup(".this")), typenod(rcvr))
	this.Left.Name.Param.Ntype = this.Right
	in := structargs(getinarg(method.Type), 1)
	out := structargs(Getoutarg(method.Type), 0)

	t := Nod(OTFUNC, nil, nil)
	l := list1(this)
	if iface != 0 && rcvr.Width < Types[Tptr].Width {
		// Building method for interface table and receiver
		// is smaller than the single pointer-sized word
		// that the interface call will pass in.
		// Add a dummy padding argument after the
		// receiver to make up the difference.
		tpad := typ(TARRAY)

		tpad.Type = Types[TUINT8]
		tpad.Bound = Types[Tptr].Width - rcvr.Width
		pad := Nod(ODCLFIELD, newname(Lookup(".pad")), typenod(tpad))
		l = list(l, pad)
	}

	t.List = concat(l, in)
	t.Rlist = out

	fn := Nod(ODCLFUNC, nil, nil)
	fn.Func.Nname = newname(newnam)
	fn.Func.Nname.Name.Defn = fn
	fn.Func.Nname.Name.Param.Ntype = t
	declare(fn.Func.Nname, PFUNC)
	funchdr(fn)

	// arg list
	var args *NodeList

	isddd := false
	for l := in; l != nil; l = l.Next {
		args = list(args, l.N.Left)
		isddd = l.N.Left.Isddd
	}

	methodrcvr := getthisx(method.Type).Type.Type

	// generate nil pointer check for better error
	if Isptr[rcvr.Etype] && rcvr.Type == methodrcvr {
		// generating wrapper from *T to T.
		n := Nod(OIF, nil, nil)

		n.Left = Nod(OEQ, this.Left, nodnil())

		// these strings are already in the reflect tables,
		// so no space cost to use them here.
		var l *NodeList

		var v Val
		v.U = rcvr.Type.Sym.Pkg.Name // package name
		l = list(l, nodlit(v))
		v.U = rcvr.Type.Sym.Name // type name
		l = list(l, nodlit(v))
		v.U = method.Sym.Name
		l = list(l, nodlit(v)) // method name
		call := Nod(OCALL, syslook("panicwrap", 0), nil)
		call.List = l
		n.Nbody = list1(call)
		fn.Nbody = list(fn.Nbody, n)
	}

	dot := adddot(Nod(OXDOT, this.Left, newname(method.Sym)))

	// generate call
	if !instrumenting && Isptr[rcvr.Etype] && Isptr[methodrcvr.Etype] && method.Embedded != 0 && !isifacemethod(method.Type) {
		// generate tail call: adjust pointer receiver and jump to embedded method.
		dot = dot.Left // skip final .M
		if !Isptr[dotlist[0].field.Type.Etype] {
			dot = Nod(OADDR, dot, nil)
		}
		as := Nod(OAS, this.Left, Nod(OCONVNOP, dot, nil))
		as.Right.Type = rcvr
		fn.Nbody = list(fn.Nbody, as)
		n := Nod(ORETJMP, nil, nil)
		n.Left = newname(methodsym(method.Sym, methodrcvr, 0))
		fn.Nbody = list(fn.Nbody, n)
	} else {
		fn.Func.Wrapper = true // ignore frame for panic+recover matching
		call := Nod(OCALL, dot, nil)
		call.List = args
		call.Isddd = isddd
		if method.Type.Outtuple > 0 {
			n := Nod(ORETURN, nil, nil)
			n.List = list1(call)
			call = n
		}

		fn.Nbody = list(fn.Nbody, call)
	}

	if false && Debug['r'] != 0 {
		dumplist("genwrapper body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn

	// wrappers where T is anonymous (struct or interface) can be duplicated.
	if rcvr.Etype == TSTRUCT || rcvr.Etype == TINTER || Isptr[rcvr.Etype] && rcvr.Type.Etype == TSTRUCT {
		fn.Func.Dupok = true
	}
	typecheck(&fn, Etop)
	typechecklist(fn.Nbody, Etop)

	inlcalls(fn)
	escAnalyze([]*Node{fn}, false)

	Curfn = nil
	funccompile(fn)
}

func hashmem(t *Type) *Node {
	sym := Pkglookup("memhash", Runtimepkg)

	n := newname(sym)
	n.Class = PFUNC
	tfn := Nod(OTFUNC, nil, nil)
	tfn.List = list(tfn.List, Nod(ODCLFIELD, nil, typenod(Ptrto(t))))
	tfn.List = list(tfn.List, Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn.List = list(tfn.List, Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn.Rlist = list(tfn.Rlist, Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	typecheck(&tfn, Etype)
	n.Type = tfn.Type
	return n
}

func hashfor(t *Type) *Node {
	var sym *Sym

	a := algtype1(t, nil)
	switch a {
	case AMEM:
		Fatalf("hashfor with AMEM type")

	case AINTER:
		sym = Pkglookup("interhash", Runtimepkg)

	case ANILINTER:
		sym = Pkglookup("nilinterhash", Runtimepkg)

	case ASTRING:
		sym = Pkglookup("strhash", Runtimepkg)

	case AFLOAT32:
		sym = Pkglookup("f32hash", Runtimepkg)

	case AFLOAT64:
		sym = Pkglookup("f64hash", Runtimepkg)

	case ACPLX64:
		sym = Pkglookup("c64hash", Runtimepkg)

	case ACPLX128:
		sym = Pkglookup("c128hash", Runtimepkg)

	default:
		sym = typesymprefix(".hash", t)
	}

	n := newname(sym)
	n.Class = PFUNC
	tfn := Nod(OTFUNC, nil, nil)
	tfn.List = list(tfn.List, Nod(ODCLFIELD, nil, typenod(Ptrto(t))))
	tfn.List = list(tfn.List, Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	tfn.Rlist = list(tfn.Rlist, Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])))
	typecheck(&tfn, Etype)
	n.Type = tfn.Type
	return n
}

// Generate a helper function to compute the hash of a value of type t.
func genhash(sym *Sym, t *Type) {
	if Debug['r'] != 0 {
		fmt.Printf("genhash %v %v\n", sym, t)
	}

	lineno = 1 // less confusing than end of input
	dclcontext = PEXTERN
	markdcl()

	// func sym(p *T, h uintptr) uintptr
	fn := Nod(ODCLFUNC, nil, nil)

	fn.Func.Nname = newname(sym)
	fn.Func.Nname.Class = PFUNC
	tfn := Nod(OTFUNC, nil, nil)
	fn.Func.Nname.Name.Param.Ntype = tfn

	n := Nod(ODCLFIELD, newname(Lookup("p")), typenod(Ptrto(t)))
	tfn.List = list(tfn.List, n)
	np := n.Left
	n = Nod(ODCLFIELD, newname(Lookup("h")), typenod(Types[TUINTPTR]))
	tfn.List = list(tfn.List, n)
	nh := n.Left
	n = Nod(ODCLFIELD, nil, typenod(Types[TUINTPTR])) // return value
	tfn.Rlist = list(tfn.Rlist, n)

	funchdr(fn)
	typecheck(&fn.Func.Nname.Name.Param.Ntype, Etype)

	// genhash is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		Fatalf("genhash %v", t)

	case TARRAY:
		if Isslice(t) {
			Fatalf("genhash %v", t)
		}

		// An array of pure memory would be handled by the
		// standard algorithm, so the element type must not be
		// pure memory.
		hashel := hashfor(t.Type)

		n := Nod(ORANGE, nil, Nod(OIND, np, nil))
		ni := newname(Lookup("i"))
		ni.Type = Types[TINT]
		n.List = list1(ni)
		n.Colas = true
		colasdefn(n.List, n)
		ni = n.List.N

		// h = hashel(&p[i], h)
		call := Nod(OCALL, hashel, nil)

		nx := Nod(OINDEX, np, ni)
		nx.Bounded = true
		na := Nod(OADDR, nx, nil)
		na.Etype = 1 // no escape to heap
		call.List = list(call.List, na)
		call.List = list(call.List, nh)
		n.Nbody = list(n.Nbody, Nod(OAS, nh, call))

		fn.Nbody = list(fn.Nbody, n)

		// Walk the struct using memhash for runs of AMEM
	// and calling specific hash functions for the others.
	case TSTRUCT:
		var first *Type

		offend := int64(0)
		var size int64
		var call *Node
		var nx *Node
		var na *Node
		var hashel *Node
		for t1 := t.Type; ; t1 = t1.Down {
			if t1 != nil && algtype1(t1.Type, nil) == AMEM && !isblanksym(t1.Sym) {
				offend = t1.Width + t1.Type.Width
				if first == nil {
					first = t1
				}

				// If it's a memory field but it's padded, stop here.
				if ispaddedfield(t1, t.Width) {
					t1 = t1.Down
				} else {
					continue
				}
			}

			// Run memhash for fields up to this one.
			if first != nil {
				size = offend - first.Width // first->width is offset
				hashel = hashmem(first.Type)

				// h = hashel(&p.first, size, h)
				call = Nod(OCALL, hashel, nil)

				nx = Nod(OXDOT, np, newname(first.Sym)) // TODO: fields from other packages?
				na = Nod(OADDR, nx, nil)
				na.Etype = 1 // no escape to heap
				call.List = list(call.List, na)
				call.List = list(call.List, nh)
				call.List = list(call.List, Nodintconst(size))
				fn.Nbody = list(fn.Nbody, Nod(OAS, nh, call))

				first = nil
			}

			if t1 == nil {
				break
			}
			if isblanksym(t1.Sym) {
				continue
			}

			// Run hash for this field.
			if algtype1(t1.Type, nil) == AMEM {
				hashel = hashmem(t1.Type)

				// h = memhash(&p.t1, h, size)
				call = Nod(OCALL, hashel, nil)

				nx = Nod(OXDOT, np, newname(t1.Sym)) // TODO: fields from other packages?
				na = Nod(OADDR, nx, nil)
				na.Etype = 1 // no escape to heap
				call.List = list(call.List, na)
				call.List = list(call.List, nh)
				call.List = list(call.List, Nodintconst(t1.Type.Width))
				fn.Nbody = list(fn.Nbody, Nod(OAS, nh, call))
			} else {
				hashel = hashfor(t1.Type)

				// h = hashel(&p.t1, h)
				call = Nod(OCALL, hashel, nil)

				nx = Nod(OXDOT, np, newname(t1.Sym)) // TODO: fields from other packages?
				na = Nod(OADDR, nx, nil)
				na.Etype = 1 // no escape to heap
				call.List = list(call.List, na)
				call.List = list(call.List, nh)
				fn.Nbody = list(fn.Nbody, Nod(OAS, nh, call))
			}
		}
	}

	r := Nod(ORETURN, nil, nil)
	r.List = list(r.List, nh)
	fn.Nbody = list(fn.Nbody, r)

	if Debug['r'] != 0 {
		dumplist("genhash body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn
	fn.Func.Dupok = true
	typecheck(&fn, Etop)
	typechecklist(fn.Nbody, Etop)
	Curfn = nil

	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode := safemode

	safemode = 0
	funccompile(fn)
	safemode = old_safemode
}

// Return node for
//	if p.field != q.field { return false }
func eqfield(p *Node, q *Node, field *Node) *Node {
	nx := Nod(OXDOT, p, field)
	ny := Nod(OXDOT, q, field)
	nif := Nod(OIF, nil, nil)
	nif.Left = Nod(ONE, nx, ny)
	r := Nod(ORETURN, nil, nil)
	r.List = list(r.List, Nodbool(false))
	nif.Nbody = list(nif.Nbody, r)
	return nif
}

func eqmemfunc(size int64, type_ *Type, needsize *int) *Node {
	var fn *Node

	switch size {
	default:
		fn = syslook("memequal", 1)
		*needsize = 1

	case 1, 2, 4, 8, 16:
		buf := fmt.Sprintf("memequal%d", int(size)*8)
		fn = syslook(buf, 1)
		*needsize = 0
	}

	substArgTypes(fn, type_, type_)
	return fn
}

// Return node for
//	if !memequal(&p.field, &q.field [, size]) { return false }
func eqmem(p *Node, q *Node, field *Node, size int64) *Node {
	var needsize int

	nx := Nod(OADDR, Nod(OXDOT, p, field), nil)
	nx.Etype = 1 // does not escape
	ny := Nod(OADDR, Nod(OXDOT, q, field), nil)
	ny.Etype = 1 // does not escape
	typecheck(&nx, Erv)
	typecheck(&ny, Erv)

	call := Nod(OCALL, eqmemfunc(size, nx.Type.Type, &needsize), nil)
	call.List = list(call.List, nx)
	call.List = list(call.List, ny)
	if needsize != 0 {
		call.List = list(call.List, Nodintconst(size))
	}

	nif := Nod(OIF, nil, nil)
	nif.Left = Nod(ONOT, call, nil)
	r := Nod(ORETURN, nil, nil)
	r.List = list(r.List, Nodbool(false))
	nif.Nbody = list(nif.Nbody, r)
	return nif
}

// Generate a helper function to check equality of two values of type t.
func geneq(sym *Sym, t *Type) {
	if Debug['r'] != 0 {
		fmt.Printf("geneq %v %v\n", sym, t)
	}

	lineno = 1 // less confusing than end of input
	dclcontext = PEXTERN
	markdcl()

	// func sym(p, q *T) bool
	fn := Nod(ODCLFUNC, nil, nil)

	fn.Func.Nname = newname(sym)
	fn.Func.Nname.Class = PFUNC
	tfn := Nod(OTFUNC, nil, nil)
	fn.Func.Nname.Name.Param.Ntype = tfn

	n := Nod(ODCLFIELD, newname(Lookup("p")), typenod(Ptrto(t)))
	tfn.List = list(tfn.List, n)
	np := n.Left
	n = Nod(ODCLFIELD, newname(Lookup("q")), typenod(Ptrto(t)))
	tfn.List = list(tfn.List, n)
	nq := n.Left
	n = Nod(ODCLFIELD, nil, typenod(Types[TBOOL]))
	tfn.Rlist = list(tfn.Rlist, n)

	funchdr(fn)

	// geneq is only called for types that have equality but
	// cannot be handled by the standard algorithms,
	// so t must be either an array or a struct.
	switch t.Etype {
	default:
		Fatalf("geneq %v", t)

	case TARRAY:
		if Isslice(t) {
			Fatalf("geneq %v", t)
		}

		// An array of pure memory would be handled by the
		// standard memequal, so the element type must not be
		// pure memory.  Even if we unrolled the range loop,
		// each iteration would be a function call, so don't bother
		// unrolling.
		nrange := Nod(ORANGE, nil, Nod(OIND, np, nil))

		ni := newname(Lookup("i"))
		ni.Type = Types[TINT]
		nrange.List = list1(ni)
		nrange.Colas = true
		colasdefn(nrange.List, nrange)
		ni = nrange.List.N

		// if p[i] != q[i] { return false }
		nx := Nod(OINDEX, np, ni)

		nx.Bounded = true
		ny := Nod(OINDEX, nq, ni)
		ny.Bounded = true

		nif := Nod(OIF, nil, nil)
		nif.Left = Nod(ONE, nx, ny)
		r := Nod(ORETURN, nil, nil)
		r.List = list(r.List, Nodbool(false))
		nif.Nbody = list(nif.Nbody, r)
		nrange.Nbody = list(nrange.Nbody, nif)
		fn.Nbody = list(fn.Nbody, nrange)

		// Walk the struct using memequal for runs of AMEM
	// and calling specific equality tests for the others.
	// Skip blank-named fields.
	case TSTRUCT:
		var first *Type

		offend := int64(0)
		var size int64
		for t1 := t.Type; ; t1 = t1.Down {
			if t1 != nil && algtype1(t1.Type, nil) == AMEM && !isblanksym(t1.Sym) {
				offend = t1.Width + t1.Type.Width
				if first == nil {
					first = t1
				}

				// If it's a memory field but it's padded, stop here.
				if ispaddedfield(t1, t.Width) {
					t1 = t1.Down
				} else {
					continue
				}
			}

			// Run memequal for fields up to this one.
			// TODO(rsc): All the calls to newname are wrong for
			// cross-package unexported fields.
			if first != nil {
				if first.Down == t1 {
					fn.Nbody = list(fn.Nbody, eqfield(np, nq, newname(first.Sym)))
				} else if first.Down.Down == t1 {
					fn.Nbody = list(fn.Nbody, eqfield(np, nq, newname(first.Sym)))
					first = first.Down
					if !isblanksym(first.Sym) {
						fn.Nbody = list(fn.Nbody, eqfield(np, nq, newname(first.Sym)))
					}
				} else {
					// More than two fields: use memequal.
					size = offend - first.Width // first->width is offset
					fn.Nbody = list(fn.Nbody, eqmem(np, nq, newname(first.Sym), size))
				}

				first = nil
			}

			if t1 == nil {
				break
			}
			if isblanksym(t1.Sym) {
				continue
			}

			// Check this field, which is not just memory.
			fn.Nbody = list(fn.Nbody, eqfield(np, nq, newname(t1.Sym)))
		}
	}

	// return true
	r := Nod(ORETURN, nil, nil)

	r.List = list(r.List, Nodbool(true))
	fn.Nbody = list(fn.Nbody, r)

	if Debug['r'] != 0 {
		dumplist("geneq body", fn.Nbody)
	}

	funcbody(fn)
	Curfn = fn
	fn.Func.Dupok = true
	typecheck(&fn, Etop)
	typechecklist(fn.Nbody, Etop)
	Curfn = nil

	// Disable safemode while compiling this code: the code we
	// generate internally can refer to unsafe.Pointer.
	// In this case it can happen if we need to generate an ==
	// for a struct containing a reflect.Value, which itself has
	// an unexported field of type unsafe.Pointer.
	old_safemode := safemode

	safemode = 0
	funccompile(fn)
	safemode = old_safemode
}

func ifacelookdot(s *Sym, t *Type, followptr *bool, ignorecase int) *Type {
	*followptr = false

	if t == nil {
		return nil
	}

	var m *Type
	var i int
	var c int
	for d := 0; d < len(dotlist); d++ {
		c = adddot1(s, t, d, &m, ignorecase)
		if c > 1 {
			Yyerror("%v.%v is ambiguous", t, s)
			return nil
		}

		if c == 1 {
			for i = 0; i < d; i++ {
				if Isptr[dotlist[i].field.Type.Etype] {
					*followptr = true
					break
				}
			}

			if m.Type.Etype != TFUNC || m.Type.Thistuple == 0 {
				Yyerror("%v.%v is a field, not a method", t, s)
				return nil
			}

			return m
		}
	}

	return nil
}

func implements(t *Type, iface *Type, m **Type, samename **Type, ptr *int) bool {
	t0 := t
	if t == nil {
		return false
	}

	// if this is too slow,
	// could sort these first
	// and then do one loop.

	if t.Etype == TINTER {
		var tm *Type
		for im := iface.Type; im != nil; im = im.Down {
			for tm = t.Type; tm != nil; tm = tm.Down {
				if tm.Sym == im.Sym {
					if Eqtype(tm.Type, im.Type) {
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

	t = methtype(t, 0)
	if t != nil {
		expandmeth(t)
	}
	var tm *Type
	var imtype *Type
	var followptr bool
	var rcvr *Type
	for im := iface.Type; im != nil; im = im.Down {
		if im.Broke {
			continue
		}
		imtype = methodfunc(im.Type, nil)
		tm = ifacelookdot(im.Sym, t, &followptr, 0)
		if tm == nil || tm.Nointerface || !Eqtype(methodfunc(tm.Type, nil), imtype) {
			if tm == nil {
				tm = ifacelookdot(im.Sym, t, &followptr, 1)
			}
			*m = im
			*samename = tm
			*ptr = 0
			return false
		}

		// if pointer receiver in method,
		// the method does not exist for value types.
		rcvr = getthisx(tm.Type).Type.Type

		if Isptr[rcvr.Etype] && !Isptr[t0.Etype] && !followptr && !isifacemethod(tm.Type) {
			if false && Debug['r'] != 0 {
				Yyerror("interface pointer mismatch")
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

	et := Simtype[t.Etype]
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

func listtreecopy(l *NodeList, lineno int32) *NodeList {
	var out *NodeList
	for ; l != nil; l = l.Next {
		out = list(out, treecopy(l.N, lineno))
	}
	return out
}

func liststmt(l *NodeList) *Node {
	n := Nod(OBLOCK, nil, nil)
	n.List = l
	if l != nil {
		n.Lineno = l.N.Lineno
	}
	return n
}

// return nelem of list
func structcount(t *Type) int {
	var s Iter

	v := 0
	for t = Structfirst(&s, &t); t != nil; t = structnext(&s) {
		v++
	}
	return v
}

// return power of 2 of the constant
// operand. -1 if it is not a power of 2.
// 1000+ if it is a -(power of 2)
func powtwo(n *Node) int {
	if n == nil || n.Op != OLITERAL || n.Type == nil {
		return -1
	}
	if !Isint[n.Type.Etype] {
		return -1
	}

	v := uint64(Mpgetfix(n.Val().U.(*Mpint)))
	b := uint64(1)
	for i := 0; i < 64; i++ {
		if b == v {
			return i
		}
		b = b << 1
	}

	if !Issigned[n.Type.Etype] {
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

// return the unsigned type for
// a signed integer type.
// returns T if input is not a
// signed integer type.
func tounsigned(t *Type) *Type {
	// this is types[et+1], but not sure
	// that this relation is immutable
	switch t.Etype {
	default:
		fmt.Printf("tounsigned: unknown type %v\n", t)
		t = nil

	case TINT:
		t = Types[TUINT]

	case TINT8:
		t = Types[TUINT8]

	case TINT16:
		t = Types[TUINT16]

	case TINT32:
		t = Types[TUINT32]

	case TINT64:
		t = Types[TUINT64]
	}

	return t
}

// magic number for signed division
// see hacker's delight chapter 10
func Smagic(m *Magic) {
	var mask uint64

	m.Bad = 0
	switch m.W {
	default:
		m.Bad = 1
		return

	case 8:
		mask = 0xff

	case 16:
		mask = 0xffff

	case 32:
		mask = 0xffffffff

	case 64:
		mask = 0xffffffffffffffff
	}

	two31 := mask ^ (mask >> 1)

	p := m.W - 1
	ad := uint64(m.Sd)
	if m.Sd < 0 {
		ad = -uint64(m.Sd)
	}

	// bad denominators
	if ad == 0 || ad == 1 || ad == two31 {
		m.Bad = 1
		return
	}

	t := two31
	ad &= mask

	anc := t - 1 - t%ad
	anc &= mask

	q1 := two31 / anc
	r1 := two31 - q1*anc
	q1 &= mask
	r1 &= mask

	q2 := two31 / ad
	r2 := two31 - q2*ad
	q2 &= mask
	r2 &= mask

	var delta uint64
	for {
		p++
		q1 <<= 1
		r1 <<= 1
		q1 &= mask
		r1 &= mask
		if r1 >= anc {
			q1++
			r1 -= anc
			q1 &= mask
			r1 &= mask
		}

		q2 <<= 1
		r2 <<= 1
		q2 &= mask
		r2 &= mask
		if r2 >= ad {
			q2++
			r2 -= ad
			q2 &= mask
			r2 &= mask
		}

		delta = ad - r2
		delta &= mask
		if q1 < delta || (q1 == delta && r1 == 0) {
			continue
		}

		break
	}

	m.Sm = int64(q2 + 1)
	if uint64(m.Sm)&two31 != 0 {
		m.Sm |= ^int64(mask)
	}
	m.S = p - m.W
}

// magic number for unsigned division
// see hacker's delight chapter 10
func Umagic(m *Magic) {
	var mask uint64

	m.Bad = 0
	m.Ua = 0

	switch m.W {
	default:
		m.Bad = 1
		return

	case 8:
		mask = 0xff

	case 16:
		mask = 0xffff

	case 32:
		mask = 0xffffffff

	case 64:
		mask = 0xffffffffffffffff
	}

	two31 := mask ^ (mask >> 1)

	m.Ud &= mask
	if m.Ud == 0 || m.Ud == two31 {
		m.Bad = 1
		return
	}

	nc := mask - (-m.Ud&mask)%m.Ud
	p := m.W - 1

	q1 := two31 / nc
	r1 := two31 - q1*nc
	q1 &= mask
	r1 &= mask

	q2 := (two31 - 1) / m.Ud
	r2 := (two31 - 1) - q2*m.Ud
	q2 &= mask
	r2 &= mask

	var delta uint64
	for {
		p++
		if r1 >= nc-r1 {
			q1 <<= 1
			q1++
			r1 <<= 1
			r1 -= nc
		} else {
			q1 <<= 1
			r1 <<= 1
		}

		q1 &= mask
		r1 &= mask
		if r2+1 >= m.Ud-r2 {
			if q2 >= two31-1 {
				m.Ua = 1
			}

			q2 <<= 1
			q2++
			r2 <<= 1
			r2++
			r2 -= m.Ud
		} else {
			if q2 >= two31 {
				m.Ua = 1
			}

			q2 <<= 1
			r2 <<= 1
			r2++
		}

		q2 &= mask
		r2 &= mask

		delta = m.Ud - 1 - r2
		delta &= mask

		if p < m.W+m.W {
			if q1 < delta || (q1 == delta && r1 == 0) {
				continue
			}
		}

		break
	}

	m.Um = q2 + 1
	m.S = p - m.W
}

func ngotype(n *Node) *Sym {
	if n.Type != nil {
		return typenamesym(n.Type)
	}
	return nil
}

// Convert raw string to the prefix that will be used in the symbol
// table.  All control characters, space, '%' and '"', as well as
// non-7-bit clean bytes turn into %xx.  The period needs escaping
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

func addinit(np **Node, init *NodeList) {
	if init == nil {
		return
	}

	n := *np
	switch n.Op {
	// There may be multiple refs to this node;
	// introduce OCONVNOP to hold init list.
	case ONAME, OLITERAL:
		n = Nod(OCONVNOP, n, nil)

		n.Type = n.Left.Type
		n.Typecheck = 1
		*np = n
	}

	n.Ninit = concat(init, n.Ninit)
	n.Ullman = UINF
}

var reservedimports = []string{
	"go",
	"type",
}

func isbadimport(path string) bool {
	if strings.Contains(path, "\x00") {
		Yyerror("import path contains NUL")
		return true
	}

	for _, ri := range reservedimports {
		if path == ri {
			Yyerror("import path %q is reserved and cannot be used", path)
			return true
		}
	}

	for _, r := range path {
		if r == utf8.RuneError {
			Yyerror("import path contains invalid UTF-8 sequence: %q", path)
			return true
		}

		if r < 0x20 || r == 0x7f {
			Yyerror("import path contains control character: %q", path)
			return true
		}

		if r == '\\' {
			Yyerror("import path contains backslash; use slash: %q", path)
			return true
		}

		if unicode.IsSpace(rune(r)) {
			Yyerror("import path contains space character: %q", path)
			return true
		}

		if strings.ContainsRune("!\"#$%&'()*,:;<=>?[]^`{|}", r) {
			Yyerror("import path contains invalid character '%c': %q", r, path)
			return true
		}
	}

	return false
}

func checknil(x *Node, init **NodeList) {
	if Isinter(x.Type) {
		x = Nod(OITAB, x, nil)
		typecheck(&x, Erv)
	}

	n := Nod(OCHECKNIL, x, nil)
	n.Typecheck = 1
	*init = list(*init, n)
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

		// Array of 1 direct iface type can be direct.
	case TARRAY:
		return t.Bound == 1 && isdirectiface(t.Type)

		// Struct with 1 field of direct iface type can be direct.
	case TSTRUCT:
		return t.Type != nil && t.Type.Down == nil && isdirectiface(t.Type.Type)
	}

	return false
}

// type2IET returns "T" if t is a concrete type,
// "I" if t is an interface type, and "E" if t is an empty interface type.
// It is used to build calls to the conv* and assert* runtime routines.
func type2IET(t *Type) string {
	if isnilinter(t) {
		return "E"
	}
	if Isinter(t) {
		return "I"
	}
	return "T"
}
