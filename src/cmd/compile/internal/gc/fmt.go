// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"cmd/internal/obj"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// A FmtFlag value is a set of flags (or 0).
// They control how the Xconv functions format their values.
// See the respective function's documentation for details.
type FmtFlag int

const (
	FmtWidth    FmtFlag = 1 << iota
	FmtLeft             // "-"
	FmtSharp            // "#"
	FmtSign             // "+"
	FmtUnsigned         // "u"
	FmtShort            // "h"
	FmtLong             // "l"
	FmtComma            // ","
	FmtByte             // "hh"
	FmtBody             // for printing export bodies
)

//
// Format conversions
//	%L int		Line numbers
//
//	%E int		etype values (aka 'Kind')
//
//	%O int		Node Opcodes
//		Flags: "%#O": print go syntax. (automatic unless fmtmode == FDbg)
//
//	%J Node*	Node details
//		Flags: "%hJ" suppresses things not relevant until walk.
//
//	%V Val*		Constant values
//
//	%S Sym*		Symbols
//		Flags: +,- #: mode (see below)
//			"%hS"	unqualified identifier in any mode
//			"%hhS"  in export mode: unqualified identifier if exported, qualified if not
//
//	%T Type*	Types
//		Flags: +,- #: mode (see below)
//			'l' definition instead of name.
//			'h' omit "func" and receiver in function types
//			'u' (only in -/Sym mode) print type identifiers wit package name instead of prefix.
//
//	%N Node*	Nodes
//		Flags: +,- #: mode (see below)
//			'h' (only in +/debug mode) suppress recursion
//			'l' (only in Error mode) print "foo (type Bar)"
//
//	%H Nodes	Nodes
//		Flags: those of %N
//			','  separate items with ',' instead of ';'
//
//   In mparith2.go and mparith3.go:
//		%B Mpint*	Big integers
//		%F Mpflt*	Big floats
//
//   %S, %T and %N obey use the following flags to set the format mode:
const (
	FErr = iota
	FDbg
	_ // formerly FExp - leave gap for now just in case there's some hard-wired dependency on the const value
	FTypeId
)

var fmtmode int = FErr

var fmtpkgpfx int // %uT stickyness

var fmtbody bool

//
// E.g. for %S:	%+S %#S %-S	print an identifier properly qualified for debug/export/internal mode.
//
// The mode flags  +, - and # are sticky, meaning they persist through
// recursions of %N, %T and %S, but not the h and l flags. The u flag is
// sticky only on %T recursions and only used in %-/Sym mode.

//
// Useful format combinations:
//
//	%+N   %+H	multiline recursive debug dump of node/nodelist
//	%+hN  %+hH	non recursive debug dump
//
//	%#N   %#T	export format
//	%#lT		type definition instead of name
//	%#hT		omit"func" and receiver in function signature
//
//	%lN		"foo (type Bar)" for error messages
//
//	%-T		type identifiers
//	%-hT		type identifiers without "func" and arg names in type signatures (methodsym)
//	%-uT		type identifiers with package name instead of prefix (typesym, dcommontype, typehash)
//

func setfmode(flags *FmtFlag) (fm int, fb bool) {
	fm = fmtmode
	fb = fmtbody
	if *flags&FmtSign != 0 {
		fmtmode = FDbg
	} else if *flags&FmtSharp != 0 {
		// ignore (textual export format no longer supported)
	} else if *flags&FmtLeft != 0 {
		fmtmode = FTypeId
	}

	if *flags&FmtBody != 0 {
		fmtbody = true
	}

	*flags &^= (FmtSharp | FmtLeft | FmtSign | FmtBody)
	return
}

// Fmt "%L": Linenumbers

var goopnames = []string{
	OADDR:     "&",
	OADD:      "+",
	OADDSTR:   "+",
	OANDAND:   "&&",
	OANDNOT:   "&^",
	OAND:      "&",
	OAPPEND:   "append",
	OAS:       "=",
	OAS2:      "=",
	OBREAK:    "break",
	OCALL:     "function call", // not actual syntax
	OCAP:      "cap",
	OCASE:     "case",
	OCLOSE:    "close",
	OCOMPLEX:  "complex",
	OCOM:      "^",
	OCONTINUE: "continue",
	OCOPY:     "copy",
	ODEC:      "--",
	ODELETE:   "delete",
	ODEFER:    "defer",
	ODIV:      "/",
	OEQ:       "==",
	OFALL:     "fallthrough",
	OFOR:      "for",
	OGE:       ">=",
	OGOTO:     "goto",
	OGT:       ">",
	OIF:       "if",
	OIMAG:     "imag",
	OINC:      "++",
	OIND:      "*",
	OLEN:      "len",
	OLE:       "<=",
	OLSH:      "<<",
	OLT:       "<",
	OMAKE:     "make",
	OMINUS:    "-",
	OMOD:      "%",
	OMUL:      "*",
	ONEW:      "new",
	ONE:       "!=",
	ONOT:      "!",
	OOROR:     "||",
	OOR:       "|",
	OPANIC:    "panic",
	OPLUS:     "+",
	OPRINTN:   "println",
	OPRINT:    "print",
	ORANGE:    "range",
	OREAL:     "real",
	ORECV:     "<-",
	ORECOVER:  "recover",
	ORETURN:   "return",
	ORSH:      ">>",
	OSELECT:   "select",
	OSEND:     "<-",
	OSUB:      "-",
	OSWITCH:   "switch",
	OXOR:      "^",
	OXFALL:    "fallthrough",
}

func (o Op) String() string {
	return oconv(o, 0)
}

func (o Op) GoString() string {
	return oconv(o, FmtSharp)
}

func oconv(o Op, flag FmtFlag) string {
	if (flag&FmtSharp != 0) || fmtmode != FDbg {
		if o >= 0 && int(o) < len(goopnames) && goopnames[o] != "" {
			return goopnames[o]
		}
	}

	if o >= 0 && int(o) < len(opnames) && opnames[o] != "" {
		return opnames[o]
	}

	return fmt.Sprintf("O-%d", o)
}

var classnames = []string{
	"Pxxx",
	"PEXTERN",
	"PAUTO",
	"PAUTOHEAP",
	"PPARAM",
	"PPARAMOUT",
	"PFUNC",
}

// Node details
func jconv(n *Node, flag FmtFlag) string {
	var p printer

	c := flag & FmtShort

	if c == 0 && n.Ullman != 0 {
		p.f(" u(%d)", n.Ullman)
	}

	if c == 0 && n.Addable {
		p.f(" a(%v)", n.Addable)
	}

	if c == 0 && n.Name != nil && n.Name.Vargen != 0 {
		p.f(" g(%d)", n.Name.Vargen)
	}

	if n.Lineno != 0 {
		p.f(" l(%d)", n.Lineno)
	}

	if c == 0 && n.Xoffset != BADWIDTH {
		p.f(" x(%d%+d)", n.Xoffset, stkdelta[n])
	}

	if n.Class != 0 {
		if int(n.Class) < len(classnames) {
			p.f(" class(%s)", classnames[n.Class])
		} else {
			p.f(" class(%d?)", n.Class)
		}
	}

	if n.Colas {
		p.f(" colas(%v)", n.Colas)
	}

	if n.Name != nil && n.Name.Funcdepth != 0 {
		p.f(" f(%d)", n.Name.Funcdepth)
	}
	if n.Func != nil && n.Func.Depth != 0 {
		p.f(" ff(%d)", n.Func.Depth)
	}

	switch n.Esc {
	case EscUnknown:
		break

	case EscHeap:
		p.s(" esc(h)")

	case EscScope:
		p.s(" esc(s)")

	case EscNone:
		p.s(" esc(no)")

	case EscNever:
		if c == 0 {
			p.s(" esc(N)")
		}

	default:
		p.f(" esc(%d)", n.Esc)
	}

	if e, ok := n.Opt().(*NodeEscState); ok && e.Escloopdepth != 0 {
		p.f(" ld(%d)", e.Escloopdepth)
	}

	if c == 0 && n.Typecheck != 0 {
		p.f(" tc(%d)", n.Typecheck)
	}

	if c == 0 && n.IsStatic {
		p.s(" static")
	}

	if n.Isddd {
		p.f(" isddd(%v)", n.Isddd)
	}

	if n.Implicit {
		p.f(" implicit(%v)", n.Implicit)
	}

	if n.Embedded != 0 {
		p.f(" embedded(%d)", n.Embedded)
	}

	if n.Addrtaken {
		p.s(" addrtaken")
	}

	if n.Assigned {
		p.s(" assigned")
	}
	if n.Bounded {
		p.s(" bounded")
	}
	if n.NonNil {
		p.s(" nonnil")
	}

	if c == 0 && n.Used {
		p.f(" used(%v)", n.Used)
	}

	return p.String()
}

// Fmt "%V": Values
func vconv(v Val, flag FmtFlag) string {
	var p printer

	switch u := v.U.(type) {
	case *Mpint:
		if !u.Rune {
			if flag&FmtSharp != 0 {
				return bconv(u, FmtSharp)
			}
			return bconv(u, 0)
		}

		switch x := u.Int64(); {
		case ' ' <= x && x < utf8.RuneSelf && x != '\\' && x != '\'':
			p.f("'%c'", int(x))

		case 0 <= x && x < 1<<16:
			p.f("'\\u%04x'", uint(int(x)))

		case 0 <= x && x <= utf8.MaxRune:
			p.f("'\\U%08x'", uint64(x))

		default:
			p.f("('\\x00' + %v)", u)
		}

	case *Mpflt:
		if flag&FmtSharp != 0 {
			return fconv(u, 0)
		}
		return fconv(u, FmtSharp)

	case *Mpcplx:
		switch {
		case flag&FmtSharp != 0:
			p.f("(%v+%vi)", &u.Real, &u.Imag)

		case v.U.(*Mpcplx).Real.CmpFloat64(0) == 0:
			p.f("%vi", fconv(&u.Imag, FmtSharp))

		case v.U.(*Mpcplx).Imag.CmpFloat64(0) == 0:
			return fconv(&u.Real, FmtSharp)

		case v.U.(*Mpcplx).Imag.CmpFloat64(0) < 0:
			p.f("(%v%vi)", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp))

		default:
			p.f("(%v+%vi)", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp))
		}

	case string:
		return strconv.Quote(u)

	case bool:
		if u {
			return "true"
		}
		return "false"

	case *NilVal:
		return "nil"

	default:
		p.f("<ctype=%d>", v.Ctype())
	}

	return p.String()
}

/*
s%,%,\n%g
s%\n+%\n%g
s%^[	]*T%%g
s%,.*%%g
s%.+%	[T&]		= "&",%g
s%^	........*\]%&~%g
s%~	%%g
*/
var etnames = []string{
	Txxx:        "Txxx",
	TINT:        "INT",
	TUINT:       "UINT",
	TINT8:       "INT8",
	TUINT8:      "UINT8",
	TINT16:      "INT16",
	TUINT16:     "UINT16",
	TINT32:      "INT32",
	TUINT32:     "UINT32",
	TINT64:      "INT64",
	TUINT64:     "UINT64",
	TUINTPTR:    "UINTPTR",
	TFLOAT32:    "FLOAT32",
	TFLOAT64:    "FLOAT64",
	TCOMPLEX64:  "COMPLEX64",
	TCOMPLEX128: "COMPLEX128",
	TBOOL:       "BOOL",
	TPTR32:      "PTR32",
	TPTR64:      "PTR64",
	TFUNC:       "FUNC",
	TARRAY:      "ARRAY",
	TSLICE:      "SLICE",
	TSTRUCT:     "STRUCT",
	TCHAN:       "CHAN",
	TMAP:        "MAP",
	TINTER:      "INTER",
	TFORW:       "FORW",
	TSTRING:     "STRING",
	TUNSAFEPTR:  "TUNSAFEPTR",
	TANY:        "ANY",
	TIDEAL:      "TIDEAL",
	TNIL:        "TNIL",
	TBLANK:      "TBLANK",
	TFUNCARGS:   "TFUNCARGS",
	TCHANARGS:   "TCHANARGS",
	TINTERMETH:  "TINTERMETH",
	TDDDFIELD:   "TDDDFIELD",
}

func (et EType) String() string {
	if int(et) < len(etnames) && etnames[et] != "" {
		return etnames[et]
	}
	return fmt.Sprintf("E-%d", et)
}

// Fmt "%S": syms
func (p *printer) symfmt(s *Sym, flag FmtFlag) *printer {
	if s.Pkg != nil && flag&FmtShort == 0 {
		switch fmtmode {
		case FErr: // This is for the user
			if s.Pkg == builtinpkg || s.Pkg == localpkg {
				return p.s(s.Name)
			}

			// If the name was used by multiple packages, display the full path,
			if s.Pkg.Name != "" && numImport[s.Pkg.Name] > 1 {
				return p.f("%q.%s", s.Pkg.Path, s.Name)
			}
			return p.s(s.Pkg.Name + "." + s.Name)

		case FDbg:
			return p.s(s.Pkg.Name + "." + s.Name)

		case FTypeId:
			if flag&FmtUnsigned != 0 {
				return p.s(s.Pkg.Name + "." + s.Name) // dcommontype, typehash
			}
			return p.s(s.Pkg.Prefix + "." + s.Name) // (methodsym), typesym, weaksym
		}
	}

	if flag&FmtByte != 0 {
		// FmtByte (hh) implies FmtShort (h)
		// skip leading "type." in method name
		name := s.Name
		if i := strings.LastIndex(name, "."); i >= 0 {
			name = name[i+1:]
		}

		if fmtmode == FDbg {
			return p.f("@%q.%s", s.Pkg.Path, name)
		}

		return p.s(name)
	}

	return p.s(s.Name)
}

var basicnames = []string{
	TINT:        "int",
	TUINT:       "uint",
	TINT8:       "int8",
	TUINT8:      "uint8",
	TINT16:      "int16",
	TUINT16:     "uint16",
	TINT32:      "int32",
	TUINT32:     "uint32",
	TINT64:      "int64",
	TUINT64:     "uint64",
	TUINTPTR:    "uintptr",
	TFLOAT32:    "float32",
	TFLOAT64:    "float64",
	TCOMPLEX64:  "complex64",
	TCOMPLEX128: "complex128",
	TBOOL:       "bool",
	TANY:        "any",
	TSTRING:     "string",
	TNIL:        "nil",
	TIDEAL:      "untyped number",
	TBLANK:      "blank",
}

func typefmt(t *Type, flag FmtFlag) string {
	if t == nil {
		return "<T>"
	}

	if t == bytetype || t == runetype {
		// in %-T mode collapse rune and byte with their originals.
		if fmtmode != FTypeId {
			return sconv(t.Sym, FmtShort)
		}
		t = Types[t.Etype]
	}

	if t == errortype {
		return "error"
	}

	// Unless the 'l' flag was specified, if the type has a name, just print that name.
	if flag&FmtLong == 0 && t.Sym != nil && t != Types[t.Etype] {
		switch fmtmode {
		case FTypeId:
			if flag&FmtShort != 0 {
				if t.Vargen != 0 {
					return fmt.Sprintf("%v·%d", sconv(t.Sym, FmtShort), t.Vargen)
				}
				return sconv(t.Sym, FmtShort)
			}

			if flag&FmtUnsigned != 0 {
				return sconv(t.Sym, FmtUnsigned)
			}

			if t.Sym.Pkg == localpkg && t.Vargen != 0 {
				return fmt.Sprintf("%v·%d", t.Sym, t.Vargen)
			}
		}

		return sconv(t.Sym, 0)
	}

	if int(t.Etype) < len(basicnames) && basicnames[t.Etype] != "" {
		prefix := ""
		if fmtmode == FErr && (t == idealbool || t == idealstring) {
			prefix = "untyped "
		}
		return prefix + basicnames[t.Etype]
	}

	if fmtmode == FDbg {
		fmtmode = 0
		str := t.Etype.String() + "-" + typefmt(t, flag)
		fmtmode = FDbg
		return str
	}

	switch t.Etype {
	case TPTR32, TPTR64:
		if fmtmode == FTypeId && (flag&FmtShort != 0) {
			return "*" + Tconv(t.Elem(), FmtShort)
		}
		return "*" + t.Elem().String()

	case TARRAY:
		if t.isDDDArray() {
			return "[...]" + t.Elem().String()
		}
		return fmt.Sprintf("[%d]%v", t.NumElem(), t.Elem())

	case TSLICE:
		return "[]" + t.Elem().String()

	case TCHAN:
		switch t.ChanDir() {
		case Crecv:
			return "<-chan " + t.Elem().String()

		case Csend:
			return "chan<- " + t.Elem().String()
		}

		if t.Elem() != nil && t.Elem().IsChan() && t.Elem().Sym == nil && t.Elem().ChanDir() == Crecv {
			return "chan (" + t.Elem().String() + ")"
		}
		return "chan " + t.Elem().String()

	case TMAP:
		return "map[" + t.Key().String() + "]" + t.Val().String()

	case TINTER:
		var buf bytes.Buffer
		buf.WriteString("interface {")
		for i, f := range t.Fields().Slice() {
			if i != 0 {
				buf.WriteString(";")
			}
			buf.WriteString(" ")
			switch {
			case f.Sym == nil:
				// Check first that a symbol is defined for this type.
				// Wrong interface definitions may have types lacking a symbol.
				break
			case exportname(f.Sym.Name):
				buf.WriteString(sconv(f.Sym, FmtShort))
			default:
				buf.WriteString(sconv(f.Sym, FmtUnsigned))
			}
			buf.WriteString(Tconv(f.Type, FmtShort))
		}
		if t.NumFields() != 0 {
			buf.WriteString(" ")
		}
		buf.WriteString("}")
		return buf.String()

	case TFUNC:
		var buf bytes.Buffer
		if flag&FmtShort != 0 {
			// no leading func
		} else {
			if t.Recv() != nil {
				buf.WriteString("method")
				buf.WriteString(Tconv(t.Recvs(), 0))
				buf.WriteString(" ")
			}
			buf.WriteString("func")
		}
		buf.WriteString(Tconv(t.Params(), 0))

		switch t.Results().NumFields() {
		case 0:
			// nothing to do

		case 1:
			buf.WriteString(" ")
			buf.WriteString(Tconv(t.Results().Field(0).Type, 0)) // struct->field->field's type

		default:
			buf.WriteString(" ")
			buf.WriteString(Tconv(t.Results(), 0))
		}
		return buf.String()

	case TSTRUCT:
		if m := t.StructType().Map; m != nil {
			mt := m.MapType()
			// Format the bucket struct for map[x]y as map.bucket[x]y.
			// This avoids a recursive print that generates very long names.
			if mt.Bucket == t {
				return "map.bucket[" + m.Key().String() + "]" + m.Val().String()
			}

			if mt.Hmap == t {
				return "map.hdr[" + m.Key().String() + "]" + m.Val().String()
			}

			if mt.Hiter == t {
				return "map.iter[" + m.Key().String() + "]" + m.Val().String()
			}

			Yyerror("unknown internal map type")
		}

		var buf bytes.Buffer
		if t.IsFuncArgStruct() {
			buf.WriteString("(")
			var flag1 FmtFlag
			if fmtmode == FTypeId || fmtmode == FErr { // no argument names on function signature, and no "noescape"/"nosplit" tags
				flag1 = FmtShort
			}
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					buf.WriteString(", ")
				}
				buf.WriteString(Fldconv(f, flag1))
			}
			buf.WriteString(")")
		} else {
			buf.WriteString("struct {")
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					buf.WriteString(";")
				}
				buf.WriteString(" ")
				buf.WriteString(Fldconv(f, FmtLong))
			}
			if t.NumFields() != 0 {
				buf.WriteString(" ")
			}
			buf.WriteString("}")
		}
		return buf.String()

	case TFORW:
		if t.Sym != nil {
			return "undefined " + t.Sym.String()
		}
		return "undefined"

	case TUNSAFEPTR:
		return "unsafe.Pointer"

	case TDDDFIELD:
		return fmt.Sprintf("%v <%v> %v", t.Etype, t.Sym, t.DDDField())

	case Txxx:
		return "Txxx"
	}

	// Don't know how to handle - fall back to detailed prints.
	return fmt.Sprintf("%v <%v> %v", t.Etype, t.Sym, t.Elem())
}

// Statements which may be rendered with a simplestmt as init.
func stmtwithinit(op Op) bool {
	switch op {
	case OIF, OFOR, OSWITCH:
		return true
	}

	return false
}

func (p *printer) stmtfmt(n *Node) *printer {
	// some statements allow for an init, but at most one,
	// but we may have an arbitrary number added, eg by typecheck
	// and inlining. If it doesn't fit the syntax, emit an enclosing
	// block starting with the init statements.

	// if we can just say "for" n->ninit; ... then do so
	simpleinit := n.Ninit.Len() == 1 && n.Ninit.First().Ninit.Len() == 0 && stmtwithinit(n.Op)

	// otherwise, print the inits as separate statements
	complexinit := n.Ninit.Len() != 0 && !simpleinit && (fmtmode != FErr)

	// but if it was for if/for/switch, put in an extra surrounding block to limit the scope
	extrablock := complexinit && stmtwithinit(n.Op)

	if extrablock {
		p.s("{")
	}

	if complexinit {
		p.f(" %v; ", n.Ninit)
	}

	switch n.Op {
	case ODCL:
		p.f("var %v %v", n.Left.Sym, n.Left.Type)

	case ODCLFIELD:
		if n.Left != nil {
			p.f("%v %v", n.Left, n.Right)
		} else {
			p.s(Nconv(n.Right, 0))
		}

	// Don't export "v = <N>" initializing statements, hope they're always
	// preceded by the DCL which will be re-parsed and typechecked to reproduce
	// the "v = <N>" again.
	case OAS, OASWB:
		if n.Colas && !complexinit {
			p.f("%v := %v", n.Left, n.Right)
		} else {
			p.f("%v = %v", n.Left, n.Right)
		}

	case OASOP:
		if n.Implicit {
			if Op(n.Etype) == OADD {
				p.f("%v++", n.Left)
			} else {
				p.f("%v--", n.Left)
			}
			break
		}

		p.f("%v %#v= %v", n.Left, Op(n.Etype), n.Right)

	case OAS2:
		if n.Colas && !complexinit {
			p.f("%v := %v", hconv(n.List, FmtComma), hconv(n.Rlist, FmtComma))
			break
		}
		fallthrough

	case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		p.f("%v = %v", hconv(n.List, FmtComma), hconv(n.Rlist, FmtComma))

	case ORETURN:
		p.f("return %v", hconv(n.List, FmtComma))

	case ORETJMP:
		p.f("retjmp %v", n.Sym)

	case OPROC:
		p.f("go %v", n.Left)

	case ODEFER:
		p.f("defer %v", n.Left)

	case OIF:
		if simpleinit {
			p.f("if %v; %v { %v }", n.Ninit.First(), n.Left, n.Nbody)
		} else {
			p.f("if %v { %v }", n.Left, n.Nbody)
		}
		if n.Rlist.Len() != 0 {
			p.f(" else { %v }", n.Rlist)
		}

	case OFOR:
		if fmtmode == FErr { // TODO maybe only if FmtShort, same below
			p.s("for loop")
			break
		}

		p.s("for")
		if simpleinit {
			p.f(" %v;", n.Ninit.First())
		} else if n.Right != nil {
			p.s(" ;")
		}

		if n.Left != nil {
			p.f(" %v", n.Left)
		}

		if n.Right != nil {
			p.f("; %v", n.Right)
		} else if simpleinit {
			p.s(";")
		}

		p.f(" { %v }", n.Nbody)

	case ORANGE:
		if fmtmode == FErr {
			p.s("for loop")
			break
		}

		if n.List.Len() == 0 {
			p.f("for range %v { %v }", n.Right, n.Nbody)
			break
		}

		p.f("for %v = range %v { %v }", hconv(n.List, FmtComma), n.Right, n.Nbody)

	case OSELECT, OSWITCH:
		if fmtmode == FErr {
			p.f("%v statement", n.Op)
			break
		}

		p.s(n.Op.GoString()) // %#v
		if simpleinit {
			p.f(" %v;", n.Ninit.First())
		}
		if n.Left != nil {
			p.f(" %s ", Nconv(n.Left, 0))
		}

		p.f(" { %v }", n.List)

	case OXCASE:
		if n.List.Len() != 0 {
			p.f("case %v: %v", hconv(n.List, FmtComma), n.Nbody)
		} else {
			p.f("default: %v", n.Nbody)
		}

	case OCASE:
		if n.Left != nil {
			p.f("case %v: %v", n.Left, n.Nbody)
		} else {
			p.f("default: %v", n.Nbody)
		}

	case OBREAK,
		OCONTINUE,
		OGOTO,
		OFALL,
		OXFALL:
		if n.Left != nil {
			p.f("%#v %v", n.Op, n.Left)
		} else {
			p.s(n.Op.GoString()) // %#v
		}

	case OEMPTY:
		break

	case OLABEL:
		p.f("%v: ", n.Left)
	}

	if extrablock {
		p.s("}")
	}

	return p
}

var opprec = []int{
	OAPPEND:       8,
	OARRAYBYTESTR: 8,
	OARRAYLIT:     8,
	OARRAYRUNESTR: 8,
	OCALLFUNC:     8,
	OCALLINTER:    8,
	OCALLMETH:     8,
	OCALL:         8,
	OCAP:          8,
	OCLOSE:        8,
	OCONVIFACE:    8,
	OCONVNOP:      8,
	OCONV:         8,
	OCOPY:         8,
	ODELETE:       8,
	OGETG:         8,
	OLEN:          8,
	OLITERAL:      8,
	OMAKESLICE:    8,
	OMAKE:         8,
	OMAPLIT:       8,
	ONAME:         8,
	ONEW:          8,
	ONONAME:       8,
	OPACK:         8,
	OPANIC:        8,
	OPAREN:        8,
	OPRINTN:       8,
	OPRINT:        8,
	ORUNESTR:      8,
	OSTRARRAYBYTE: 8,
	OSTRARRAYRUNE: 8,
	OSTRUCTLIT:    8,
	OTARRAY:       8,
	OTCHAN:        8,
	OTFUNC:        8,
	OTINTER:       8,
	OTMAP:         8,
	OTSTRUCT:      8,
	OINDEXMAP:     8,
	OINDEX:        8,
	OSLICE:        8,
	OSLICESTR:     8,
	OSLICEARR:     8,
	OSLICE3:       8,
	OSLICE3ARR:    8,
	ODOTINTER:     8,
	ODOTMETH:      8,
	ODOTPTR:       8,
	ODOTTYPE2:     8,
	ODOTTYPE:      8,
	ODOT:          8,
	OXDOT:         8,
	OCALLPART:     8,
	OPLUS:         7,
	ONOT:          7,
	OCOM:          7,
	OMINUS:        7,
	OADDR:         7,
	OIND:          7,
	ORECV:         7,
	OMUL:          6,
	ODIV:          6,
	OMOD:          6,
	OLSH:          6,
	ORSH:          6,
	OAND:          6,
	OANDNOT:       6,
	OADD:          5,
	OSUB:          5,
	OOR:           5,
	OXOR:          5,
	OEQ:           4,
	OLT:           4,
	OLE:           4,
	OGE:           4,
	OGT:           4,
	ONE:           4,
	OCMPSTR:       4,
	OCMPIFACE:     4,
	OSEND:         3,
	OANDAND:       2,
	OOROR:         1,
	// Statements handled by stmtfmt
	OAS:         -1,
	OAS2:        -1,
	OAS2DOTTYPE: -1,
	OAS2FUNC:    -1,
	OAS2MAPR:    -1,
	OAS2RECV:    -1,
	OASOP:       -1,
	OBREAK:      -1,
	OCASE:       -1,
	OCONTINUE:   -1,
	ODCL:        -1,
	ODCLFIELD:   -1,
	ODEFER:      -1,
	OEMPTY:      -1,
	OFALL:       -1,
	OFOR:        -1,
	OGOTO:       -1,
	OIF:         -1,
	OLABEL:      -1,
	OPROC:       -1,
	ORANGE:      -1,
	ORETURN:     -1,
	OSELECT:     -1,
	OSWITCH:     -1,
	OXCASE:      -1,
	OXFALL:      -1,
	OEND:        0,
}

func (p *printer) exprfmt(n *Node, prec int) *printer {
	for n != nil && n.Implicit && (n.Op == OIND || n.Op == OADDR) {
		n = n.Left
	}

	if n == nil {
		return p.s("<N>")
	}

	nprec := opprec[n.Op]
	if n.Op == OTYPE && n.Sym != nil {
		nprec = 8
	}

	if prec > nprec {
		return p.f("(%v)", n)
	}

	switch n.Op {
	case OPAREN:
		return p.f("(%v)", n.Left)

	case ODDDARG:
		return p.s("... argument")

	case OREGISTER:
		return p.s(obj.Rconv(int(n.Reg)))

	case OLITERAL: // this is a bit of a mess
		if fmtmode == FErr {
			if n.Orig != nil && n.Orig != n {
				return p.exprfmt(n.Orig, prec)
			}
			if n.Sym != nil {
				return p.s(sconv(n.Sym, 0))
			}
		}
		if n.Val().Ctype() == CTNIL && n.Orig != nil && n.Orig != n {
			return p.exprfmt(n.Orig, prec)
		}
		if n.Type != nil && n.Type.Etype != TIDEAL && n.Type.Etype != TNIL && n.Type != idealbool && n.Type != idealstring {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if n.Type.IsPtr() || (n.Type.IsChan() && n.Type.ChanDir() == Crecv) {
				return p.f("(%v)(%v)", n.Type, vconv(n.Val(), 0))
			} else {
				return p.f("%v(%v)", n.Type, vconv(n.Val(), 0))
			}
		}

		return p.s(vconv(n.Val(), 0))

	// Special case: name used as local variable in export.
	// _ becomes ~b%d internally; print as _ for export
	case ONAME:
		if fmtmode == FErr && n.Sym != nil && n.Sym.Name[0] == '~' && n.Sym.Name[1] == 'b' {
			return p.s("_")
		}
		fallthrough

	case OPACK, ONONAME:
		return p.s(sconv(n.Sym, 0))

	case OTYPE:
		if n.Type == nil && n.Sym != nil {
			return p.s(sconv(n.Sym, 0))
		}
		return p.s(Tconv(n.Type, 0))

	case OTARRAY:
		if n.Left != nil {
			return p.f("[]%v", n.Left)
		}
		return p.f("[]%v", n.Right) // happens before typecheck

	case OTMAP:
		return p.f("map[%v]%v", n.Left, n.Right)

	case OTCHAN:
		switch ChanDir(n.Etype) {
		case Crecv:
			return p.f("<-chan %v", n.Left)

		case Csend:
			return p.f("chan<- %v", n.Left)

		default:
			if n.Left != nil && n.Left.Op == OTCHAN && n.Left.Sym == nil && ChanDir(n.Left.Etype) == Crecv {
				return p.f("chan (%v)", n.Left)
			} else {
				return p.f("chan %v", n.Left)
			}
		}

	case OTSTRUCT:
		return p.s("<struct>")

	case OTINTER:
		return p.s("<inter>")

	case OTFUNC:
		return p.s("<func>")

	case OCLOSURE:
		if fmtmode == FErr {
			return p.s("func literal")
		}
		if n.Nbody.Len() != 0 {
			return p.f("%v { %v }", n.Type, n.Nbody)
		}
		return p.f("%v { %v }", n.Type, n.Func.Closure.Nbody)

	case OCOMPLIT:
		ptrlit := n.Right != nil && n.Right.Implicit && n.Right.Type != nil && n.Right.Type.IsPtr()
		if fmtmode == FErr {
			if n.Right != nil && n.Right.Type != nil && !n.Implicit {
				if ptrlit {
					return p.f("&%v literal", n.Right.Type.Elem())
				} else {
					return p.f("%v literal", n.Right.Type)
				}
			}

			return p.s("composite literal")
		}

		return p.f("(%v{ %v })", n.Right, hconv(n.List, FmtComma))

	case OPTRLIT:
		return p.f("&%v", n.Left)

	case OSTRUCTLIT, OARRAYLIT, OMAPLIT:
		if fmtmode == FErr {
			return p.f("%v literal", n.Type)
		}
		return p.f("(%v{ %v })", n.Type, hconv(n.List, FmtComma))

	case OKEY:
		if n.Left != nil && n.Right != nil {
			return p.f("%v:%v", n.Left, n.Right)
		}

		if n.Left == nil && n.Right != nil {
			return p.f(":%v", n.Right)
		}
		if n.Left != nil && n.Right == nil {
			return p.f("%v:", n.Left)
		}
		return p.s(":")

	case OCALLPART:
		p.exprfmt(n.Left, nprec)
		if n.Right == nil || n.Right.Sym == nil {
			return p.s(".<nil>")
		}
		return p.f(".%v", sconv(n.Right.Sym, FmtShort|FmtByte))

	case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
		p.exprfmt(n.Left, nprec)
		if n.Sym == nil {
			return p.s(".<nil>")
		}
		return p.f(".%v", sconv(n.Sym, FmtShort|FmtByte))

	case ODOTTYPE, ODOTTYPE2:
		p.exprfmt(n.Left, nprec)
		if n.Right != nil {
			return p.f(".(%v)", n.Right)
		}
		return p.f(".(%v)", n.Type)

	case OINDEX, OINDEXMAP:
		return p.exprfmt(n.Left, nprec).f("[%v]", n.Right)

	case OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
		p.exprfmt(n.Left, nprec)
		p.s("[")
		low, high, max := n.SliceBounds()
		if low != nil {
			p.s(low.String())
		}
		p.s(":")
		if high != nil {
			p.s(high.String())
		}
		if n.Op.IsSlice3() {
			p.s(":")
			if max != nil {
				p.s(max.String())
			}
		}
		return p.s("]")

	case OCOPY, OCOMPLEX:
		return p.f("%#v(%v, %v)", n.Op, n.Left, n.Right)

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		ORUNESTR:
		if n.Type == nil || n.Type.Sym == nil {
			return p.f("(%v)(%v)", n.Type, n.Left)
		}
		if n.Left != nil {
			return p.f("%v(%v)", n.Type, n.Left)
		}
		return p.f("%v(%v)", n.Type, hconv(n.List, FmtComma))

	case OREAL,
		OIMAG,
		OAPPEND,
		OCAP,
		OCLOSE,
		ODELETE,
		OLEN,
		OMAKE,
		ONEW,
		OPANIC,
		ORECOVER,
		OPRINT,
		OPRINTN:
		if n.Left != nil {
			return p.f("%#v(%v)", n.Op, n.Left)
		}
		if n.Isddd {
			return p.f("%#v(%v...)", n.Op, hconv(n.List, FmtComma))
		}
		return p.f("%#v(%v)", n.Op, hconv(n.List, FmtComma))

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		p.exprfmt(n.Left, nprec)
		if n.Isddd {
			return p.f("(%v...)", hconv(n.List, FmtComma))
		}
		return p.f("(%v)", hconv(n.List, FmtComma))

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		if n.List.Len() != 0 { // pre-typecheck
			return p.f("make(%v, %v)", n.Type, hconv(n.List, FmtComma))
		}
		if n.Right != nil {
			return p.f("make(%v, %v, %v)", n.Type, n.Left, n.Right)
		}
		if n.Left != nil && (n.Op == OMAKESLICE || !n.Left.Type.IsUntyped()) {
			return p.f("make(%v, %v)", n.Type, n.Left)
		}
		return p.f("make(%v)", n.Type)

		// Unary
	case OPLUS,
		OMINUS,
		OADDR,
		OCOM,
		OIND,
		ONOT,
		ORECV:
		p.s(n.Op.GoString()) // %#v
		if n.Left.Op == n.Op {
			p.s(" ")
		}
		return p.exprfmt(n.Left, nprec+1)

		// Binary
	case OADD,
		OAND,
		OANDAND,
		OANDNOT,
		ODIV,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLT,
		OLSH,
		OMOD,
		OMUL,
		ONE,
		OOR,
		OOROR,
		ORSH,
		OSEND,
		OSUB,
		OXOR:
		p.exprfmt(n.Left, nprec)
		p.f(" %#v ", n.Op)
		p.exprfmt(n.Right, nprec+1)
		return p

	case OADDSTR:
		i := 0
		for _, n1 := range n.List.Slice() {
			if i != 0 {
				p.s(" + ")
			}
			p.exprfmt(n1, nprec)
			i++
		}
		return p

	case OCMPSTR, OCMPIFACE:
		p.exprfmt(n.Left, nprec)
		// TODO(marvin): Fix Node.EType type union.
		p.f(" %#v ", Op(n.Etype))
		p.exprfmt(n.Right, nprec+1)
		return p

	case ODCLCONST:
		// if exporting, DCLCONST should just be removed as its usage
		// has already been replaced with literals
		if fmtbody {
			return p.s("")
		}
	}

	return p.f("<node %v>", n.Op)
}

func (p *printer) nodefmt(n *Node, flag FmtFlag) *printer {
	t := n.Type

	// we almost always want the original, except in export mode for literals
	// this saves the importer some work, and avoids us having to redo some
	// special casing for package unsafe
	if n.Op != OLITERAL && n.Orig != nil {
		n = n.Orig
	}

	if flag&FmtLong != 0 && t != nil {
		if t.Etype == TNIL {
			return p.s("nil")
		} else {
			return p.f("%v (type %v)", n, t)
		}
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if opprec[n.Op] < 0 {
		return p.stmtfmt(n)
	}

	return p.exprfmt(n, 0)
}

func (p *printer) nodedump(n *Node, flag FmtFlag) *printer {
	if n == nil {
		return p
	}

	recur := flag&FmtShort == 0

	if recur {
		p.indent()
		if dumpdepth > 10 {
			return p.s("...")
		}

		if n.Ninit.Len() != 0 {
			p.f("%v-init%v", n.Op, n.Ninit)
			p.indent()
		}
	}

	switch n.Op {
	default:
		p.f("%v%v", n.Op, jconv(n, 0))

	case OREGISTER, OINDREG:
		p.f("%v-%v%v", n.Op, obj.Rconv(int(n.Reg)), jconv(n, 0))

	case OLITERAL:
		p.f("%v-%v%v", n.Op, vconv(n.Val(), 0), jconv(n, 0))

	case ONAME, ONONAME:
		if n.Sym != nil {
			p.f("%v-%v%v", n.Op, n.Sym, jconv(n, 0))
		} else {
			p.f("%v%v", n.Op, jconv(n, 0))
		}
		if recur && n.Type == nil && n.Name != nil && n.Name.Param != nil && n.Name.Param.Ntype != nil {
			p.indent()
			p.f("%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}

	case OASOP:
		p.f("%v-%v%v", n.Op, Op(n.Etype), jconv(n, 0))

	case OTYPE:
		p.f("%v %v%v type=%v", n.Op, n.Sym, jconv(n, 0), n.Type)
		if recur && n.Type == nil && n.Name.Param.Ntype != nil {
			p.indent()
			p.f("%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}
	}

	if n.Sym != nil && n.Op != ONAME {
		p.f(" %v", n.Sym)
	}

	if n.Type != nil {
		p.f(" %v", n.Type)
	}

	if recur {
		if n.Left != nil {
			p.s(Nconv(n.Left, 0))
		}
		if n.Right != nil {
			p.s(Nconv(n.Right, 0))
		}
		if n.List.Len() != 0 {
			p.indent()
			p.f("%v-list%v", n.Op, n.List)
		}

		if n.Rlist.Len() != 0 {
			p.indent()
			p.f("%v-rlist%v", n.Op, n.Rlist)
		}

		if n.Nbody.Len() != 0 {
			p.indent()
			p.f("%v-body%v", n.Op, n.Nbody)
		}
	}

	return p
}

func (s *Sym) String() string {
	return sconv(s, 0)
}

// Fmt "%S": syms
// Flags:  "%hS" suppresses qualifying with package
func sconv(s *Sym, flag FmtFlag) string {
	var p printer

	if flag&FmtLong != 0 {
		panic("linksymfmt")
	}

	if s == nil {
		return "<S>"
	}

	if s.Name == "_" {
		return "_"
	}

	sf := flag
	sm, sb := setfmode(&flag)
	p.symfmt(s, flag)
	flag = sf
	fmtmode = sm
	fmtbody = sb

	return p.String()
}

func (t *Type) String() string {
	return Tconv(t, 0)
}

func Fldconv(f *Field, flag FmtFlag) string {
	if f == nil {
		return "<T>"
	}

	sf := flag
	sm, sb := setfmode(&flag)

	if fmtmode == FTypeId && (sf&FmtUnsigned != 0) {
		fmtpkgpfx++
	}
	if fmtpkgpfx != 0 {
		flag |= FmtUnsigned
	}

	var name string
	if flag&FmtShort == 0 {
		s := f.Sym

		// Take the name from the original, lest we substituted it with ~r%d or ~b%d.
		// ~r%d is a (formerly) unnamed result.
		if fmtmode == FErr && f.Nname != nil {
			if f.Nname.Orig != nil {
				s = f.Nname.Orig.Sym
				if s != nil && s.Name[0] == '~' {
					if s.Name[1] == 'r' { // originally an unnamed result
						s = nil
					} else if s.Name[1] == 'b' { // originally the blank identifier _
						s = Lookup("_")
					}
				}
			} else {
				s = nil
			}
		}

		if s != nil && f.Embedded == 0 {
			if f.Funarg != FunargNone {
				name = Nconv(f.Nname, 0)
			} else if flag&FmtLong != 0 {
				name = sconv(s, FmtShort|FmtByte)
				if !exportname(name) && flag&FmtUnsigned == 0 {
					name = sconv(s, 0) // qualify non-exported names (used on structs, not on funarg)
				}
			} else {
				name = sconv(s, 0)
			}
		}
	}

	var typ string
	if f.Isddd {
		typ = "..." + Tconv(f.Type.Elem(), 0)
	} else {
		typ = Tconv(f.Type, 0)
	}

	str := typ
	if name != "" {
		str = name + " " + typ
	}

	// The fmtbody flag is intended to suppress escape analysis annotations
	// when printing a function type used in a function body.
	// (The escape analysis tags do not apply to func vars.)
	// But it must not suppress struct field tags.
	// See golang.org/issue/13777 and golang.org/issue/14331.
	if flag&FmtShort == 0 && (!fmtbody || f.Funarg == FunargNone) && f.Note != "" {
		str += " " + strconv.Quote(f.Note)
	}

	if fmtmode == FTypeId && (sf&FmtUnsigned != 0) {
		fmtpkgpfx--
	}

	flag = sf
	fmtbody = sb
	fmtmode = sm
	return str
}

// Fmt "%T": types.
// Flags: 'l' print definition, not name
//	  'h' omit 'func' and receiver from function types, short type names
//	  'u' package name, not prefix (FTypeId mode, sticky)
func Tconv(t *Type, flag FmtFlag) string {
	if t == nil {
		return "<T>"
	}

	if t.Trecur > 4 {
		return "<...>"
	}

	t.Trecur++
	sf := flag
	sm, sb := setfmode(&flag)

	if fmtmode == FTypeId && (sf&FmtUnsigned != 0) {
		fmtpkgpfx++
	}
	if fmtpkgpfx != 0 {
		flag |= FmtUnsigned
	}

	str := typefmt(t, flag)

	if fmtmode == FTypeId && (sf&FmtUnsigned != 0) {
		fmtpkgpfx--
	}

	flag = sf
	fmtbody = sb
	fmtmode = sm
	t.Trecur--
	return str
}

func (n *Node) String() string {
	return Nconv(n, 0)
}

// Fmt '%N': Nodes.
// Flags: 'l' suffix with "(type %T)" where possible
//	  '+h' in debug mode, don't recurse, no multiline output
func Nconv(n *Node, flag FmtFlag) string {
	var p printer

	if n == nil {
		return "<N>"
	}
	sf := flag
	sm, sb := setfmode(&flag)

	switch fmtmode {
	case FErr:
		p.nodefmt(n, flag)

	case FDbg:
		dumpdepth++
		p.nodedump(n, flag)
		dumpdepth--

	default:
		Fatalf("unhandled %%N mode")
	}

	flag = sf
	fmtbody = sb
	fmtmode = sm

	return p.String()
}

func (n Nodes) String() string {
	return hconv(n, 0)
}

// Fmt '%H': Nodes.
// Flags: all those of %N plus ',': separate with comma's instead of semicolons.
func hconv(l Nodes, flag FmtFlag) string {
	var p printer

	if l.Len() == 0 && fmtmode == FDbg {
		return "<nil>"
	}

	sf := flag
	sm, sb := setfmode(&flag)
	sep := "; "
	if fmtmode == FDbg {
		sep = "\n"
	} else if flag&FmtComma != 0 {
		sep = ", "
	}

	for i, n := range l.Slice() {
		p.s(Nconv(n, 0))
		if i+1 < l.Len() {
			p.s(sep)
		}
	}

	flag = sf
	fmtbody = sb
	fmtmode = sm

	return p.String()
}

func dumplist(s string, l Nodes) {
	fmt.Printf("%s%v\n", s, hconv(l, FmtSign))
}

func Dump(s string, n *Node) {
	fmt.Printf("%s [%p]%v\n", s, n, Nconv(n, FmtSign))
}

// printer is a buffer for creating longer formatted strings.
type printer struct {
	buf []byte
}

// printer implements io.Writer.
func (p *printer) Write(buf []byte) (n int, err error) {
	p.buf = append(p.buf, buf...)
	return len(buf), nil
}

// printer implements the Stringer interface.
func (p *printer) String() string {
	return string(p.buf)
}

// s prints the string s to p and returns p.
func (p *printer) s(s string) *printer {
	p.buf = append(p.buf, s...)
	return p
}

// f prints the formatted arguments to p and returns p.
func (p *printer) f(format string, args ...interface{}) *printer {
	fmt.Fprintf(p, format, args...)
	return p
}

// TODO(gri) make this a field of printer
var dumpdepth int

// indent prints indentation to p.
func (p *printer) indent() {
	p.s("\n")
	for i := 0; i < dumpdepth; i++ {
		p.s(".   ")
	}
}
