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
	FExp
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
		fmtmode = FExp
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

// Fmt "%J": Node details.
func jconv(n *Node, flag FmtFlag) string {
	var buf bytes.Buffer

	c := flag & FmtShort

	if c == 0 && n.Ullman != 0 {
		fmt.Fprintf(&buf, " u(%d)", n.Ullman)
	}

	if c == 0 && n.Addable {
		fmt.Fprintf(&buf, " a(%v)", n.Addable)
	}

	if c == 0 && n.Name != nil && n.Name.Vargen != 0 {
		fmt.Fprintf(&buf, " g(%d)", n.Name.Vargen)
	}

	if n.Lineno != 0 {
		fmt.Fprintf(&buf, " l(%d)", n.Lineno)
	}

	if c == 0 && n.Xoffset != BADWIDTH {
		fmt.Fprintf(&buf, " x(%d%+d)", n.Xoffset, stkdelta[n])
	}

	if n.Class != 0 {
		if int(n.Class) < len(classnames) {
			fmt.Fprintf(&buf, " class(%s)", classnames[n.Class])
		} else {
			fmt.Fprintf(&buf, " class(%d?)", n.Class)
		}
	}

	if n.Colas {
		fmt.Fprintf(&buf, " colas(%v)", n.Colas)
	}

	if n.Name != nil && n.Name.Funcdepth != 0 {
		fmt.Fprintf(&buf, " f(%d)", n.Name.Funcdepth)
	}
	if n.Func != nil && n.Func.Depth != 0 {
		fmt.Fprintf(&buf, " ff(%d)", n.Func.Depth)
	}

	switch n.Esc {
	case EscUnknown:
		break

	case EscHeap:
		buf.WriteString(" esc(h)")

	case EscScope:
		buf.WriteString(" esc(s)")

	case EscNone:
		buf.WriteString(" esc(no)")

	case EscNever:
		if c == 0 {
			buf.WriteString(" esc(N)")
		}

	default:
		fmt.Fprintf(&buf, " esc(%d)", n.Esc)
	}

	if e, ok := n.Opt().(*NodeEscState); ok && e.Escloopdepth != 0 {
		fmt.Fprintf(&buf, " ld(%d)", e.Escloopdepth)
	}

	if c == 0 && n.Typecheck != 0 {
		fmt.Fprintf(&buf, " tc(%d)", n.Typecheck)
	}

	if c == 0 && n.Dodata != 0 {
		fmt.Fprintf(&buf, " dd(%d)", n.Dodata)
	}

	if n.Isddd {
		fmt.Fprintf(&buf, " isddd(%v)", n.Isddd)
	}

	if n.Implicit {
		fmt.Fprintf(&buf, " implicit(%v)", n.Implicit)
	}

	if n.Embedded != 0 {
		fmt.Fprintf(&buf, " embedded(%d)", n.Embedded)
	}

	if n.Addrtaken {
		buf.WriteString(" addrtaken")
	}

	if n.Assigned {
		buf.WriteString(" assigned")
	}
	if n.Bounded {
		buf.WriteString(" bounded")
	}
	if n.NonNil {
		buf.WriteString(" nonnil")
	}

	if c == 0 && n.Used {
		fmt.Fprintf(&buf, " used(%v)", n.Used)
	}
	return buf.String()
}

// Fmt "%V": Values
func vconv(v Val, flag FmtFlag) string {
	switch u := v.U.(type) {
	case *Mpint:
		if !u.Rune {
			if (flag&FmtSharp != 0) || fmtmode == FExp {
				return bconv(u, FmtSharp)
			}
			return bconv(u, 0)
		}

		x := u.Int64()
		if ' ' <= x && x < utf8.RuneSelf && x != '\\' && x != '\'' {
			return fmt.Sprintf("'%c'", int(x))
		}
		if 0 <= x && x < 1<<16 {
			return fmt.Sprintf("'\\u%04x'", uint(int(x)))
		}
		if 0 <= x && x <= utf8.MaxRune {
			return fmt.Sprintf("'\\U%08x'", uint64(x))
		}
		return fmt.Sprintf("('\\x00' + %v)", u)

	case *Mpflt:
		if (flag&FmtSharp != 0) || fmtmode == FExp {
			return fconv(u, 0)
		}
		return fconv(u, FmtSharp)

	case *Mpcplx:
		if (flag&FmtSharp != 0) || fmtmode == FExp {
			return fmt.Sprintf("(%v+%vi)", &u.Real, &u.Imag)
		}
		if v.U.(*Mpcplx).Real.CmpFloat64(0) == 0 {
			return fmt.Sprintf("%vi", fconv(&u.Imag, FmtSharp))
		}
		if v.U.(*Mpcplx).Imag.CmpFloat64(0) == 0 {
			return fconv(&u.Real, FmtSharp)
		}
		if v.U.(*Mpcplx).Imag.CmpFloat64(0) < 0 {
			return fmt.Sprintf("(%v%vi)", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp))
		}
		return fmt.Sprintf("(%v+%vi)", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp))

	case string:
		return strconv.Quote(u)

	case bool:
		if u {
			return "true"
		}
		return "false"

	case *NilVal:
		return "nil"
	}

	return fmt.Sprintf("<ctype=%d>", v.Ctype())
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
func symfmt(s *Sym, flag FmtFlag) string {
	if s.Pkg != nil && flag&FmtShort == 0 {
		switch fmtmode {
		case FErr: // This is for the user
			if s.Pkg == builtinpkg || s.Pkg == localpkg {
				return s.Name
			}

			// If the name was used by multiple packages, display the full path,
			if s.Pkg.Name != "" && numImport[s.Pkg.Name] > 1 {
				return fmt.Sprintf("%q.%s", s.Pkg.Path, s.Name)
			}
			return s.Pkg.Name + "." + s.Name

		case FDbg:
			return s.Pkg.Name + "." + s.Name

		case FTypeId:
			if flag&FmtUnsigned != 0 {
				return s.Pkg.Name + "." + s.Name // dcommontype, typehash
			}
			return s.Pkg.Prefix + "." + s.Name // (methodsym), typesym, weaksym

		case FExp:
			if s.Name != "" && s.Name[0] == '.' {
				Fatalf("exporting synthetic symbol %s", s.Name)
			}
			if s.Pkg != builtinpkg {
				return fmt.Sprintf("@%q.%s", s.Pkg.Path, s.Name)
			}
		}
	}

	if flag&FmtByte != 0 {
		// FmtByte (hh) implies FmtShort (h)
		// skip leading "type." in method name
		p := s.Name
		if i := strings.LastIndex(s.Name, "."); i >= 0 {
			p = s.Name[i+1:]
		}

		// exportname needs to see the name without the prefix too.
		if (fmtmode == FExp && !exportname(p)) || fmtmode == FDbg {
			return fmt.Sprintf("@%q.%s", s.Pkg.Path, p)
		}

		return p
	}

	return s.Name
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
			fallthrough

		case FExp:
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
			break

		case 1:
			if fmtmode != FExp {
				buf.WriteString(" ")
				buf.WriteString(Tconv(t.Results().Field(0).Type, 0)) // struct->field->field's type
				break
			}
			fallthrough

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
		if fmtmode == FExp {
			return "@\"unsafe\".Pointer"
		}
		return "unsafe.Pointer"

	case TDDDFIELD:
		if fmtmode == FExp {
			Fatalf("cannot use TDDDFIELD with old exporter")
		}
		return fmt.Sprintf("%v <%v> %v", t.Etype, t.Sym, t.DDDField())

	case Txxx:
		return "Txxx"
	}

	if fmtmode == FExp {
		Fatalf("missing %v case during export", t.Etype)
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

func stmtfmt(n *Node) string {
	var f string

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
		f += "{"
	}

	if complexinit {
		f += fmt.Sprintf(" %v; ", n.Ninit)
	}

	switch n.Op {
	case ODCL:
		if fmtmode == FExp {
			switch n.Left.Class {
			case PPARAM, PPARAMOUT, PAUTO, PAUTOHEAP:
				f += fmt.Sprintf("var %v %v", n.Left, n.Left.Type)
				goto ret
			}
		}

		f += fmt.Sprintf("var %v %v", n.Left.Sym, n.Left.Type)

	case ODCLFIELD:
		if n.Left != nil {
			f += fmt.Sprintf("%v %v", n.Left, n.Right)
		} else {
			f += Nconv(n.Right, 0)
		}

	// Don't export "v = <N>" initializing statements, hope they're always
	// preceded by the DCL which will be re-parsed and typechecked to reproduce
	// the "v = <N>" again.
	case OAS, OASWB:
		if fmtmode == FExp && n.Right == nil {
			break
		}

		if n.Colas && !complexinit {
			f += fmt.Sprintf("%v := %v", n.Left, n.Right)
		} else {
			f += fmt.Sprintf("%v = %v", n.Left, n.Right)
		}

	case OASOP:
		if n.Implicit {
			if Op(n.Etype) == OADD {
				f += fmt.Sprintf("%v++", n.Left)
			} else {
				f += fmt.Sprintf("%v--", n.Left)
			}
			break
		}

		f += fmt.Sprintf("%v %#v= %v", n.Left, Op(n.Etype), n.Right)

	case OAS2:
		if n.Colas && !complexinit {
			f += fmt.Sprintf("%v := %v", hconv(n.List, FmtComma), hconv(n.Rlist, FmtComma))
			break
		}
		fallthrough

	case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		f += fmt.Sprintf("%v = %v", hconv(n.List, FmtComma), hconv(n.Rlist, FmtComma))

	case ORETURN:
		f += fmt.Sprintf("return %v", hconv(n.List, FmtComma))

	case ORETJMP:
		f += fmt.Sprintf("retjmp %v", n.Sym)

	case OPROC:
		f += fmt.Sprintf("go %v", n.Left)

	case ODEFER:
		f += fmt.Sprintf("defer %v", n.Left)

	case OIF:
		if simpleinit {
			f += fmt.Sprintf("if %v; %v { %v }", n.Ninit.First(), n.Left, n.Nbody)
		} else {
			f += fmt.Sprintf("if %v { %v }", n.Left, n.Nbody)
		}
		if n.Rlist.Len() != 0 {
			f += fmt.Sprintf(" else { %v }", n.Rlist)
		}

	case OFOR:
		if fmtmode == FErr { // TODO maybe only if FmtShort, same below
			f += "for loop"
			break
		}

		f += "for"
		if simpleinit {
			f += fmt.Sprintf(" %v;", n.Ninit.First())
		} else if n.Right != nil {
			f += " ;"
		}

		if n.Left != nil {
			f += fmt.Sprintf(" %v", n.Left)
		}

		if n.Right != nil {
			f += fmt.Sprintf("; %v", n.Right)
		} else if simpleinit {
			f += ";"
		}

		f += fmt.Sprintf(" { %v }", n.Nbody)

	case ORANGE:
		if fmtmode == FErr {
			f += "for loop"
			break
		}

		if n.List.Len() == 0 {
			f += fmt.Sprintf("for range %v { %v }", n.Right, n.Nbody)
			break
		}

		f += fmt.Sprintf("for %v = range %v { %v }", hconv(n.List, FmtComma), n.Right, n.Nbody)

	case OSELECT, OSWITCH:
		if fmtmode == FErr {
			f += fmt.Sprintf("%v statement", n.Op)
			break
		}

		f += n.Op.GoString() // %#v
		if simpleinit {
			f += fmt.Sprintf(" %v;", n.Ninit.First())
		}
		if n.Left != nil {
			f += fmt.Sprintf(" %s ", Nconv(n.Left, 0))
		}

		f += fmt.Sprintf(" { %v }", n.List)

	case OCASE, OXCASE:
		if n.List.Len() != 0 {
			f += fmt.Sprintf("case %v: %v", hconv(n.List, FmtComma), n.Nbody)
		} else {
			f += fmt.Sprintf("default: %v", n.Nbody)
		}

	case OBREAK,
		OCONTINUE,
		OGOTO,
		OFALL,
		OXFALL:
		if n.Left != nil {
			f += fmt.Sprintf("%#v %v", n.Op, n.Left)
		} else {
			f += n.Op.GoString() // %#v
		}

	case OEMPTY:
		break

	case OLABEL:
		f += fmt.Sprintf("%v: ", n.Left)
	}

ret:
	if extrablock {
		f += "}"
	}

	return f
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

func exprfmt(n *Node, prec int) string {
	for n != nil && n.Implicit && (n.Op == OIND || n.Op == OADDR) {
		n = n.Left
	}

	if n == nil {
		return "<N>"
	}

	nprec := opprec[n.Op]
	if n.Op == OTYPE && n.Sym != nil {
		nprec = 8
	}

	if prec > nprec {
		return fmt.Sprintf("(%v)", n)
	}

	switch n.Op {
	case OPAREN:
		return fmt.Sprintf("(%v)", n.Left)

	case ODDDARG:
		return "... argument"

	case OREGISTER:
		return obj.Rconv(int(n.Reg))

	case OLITERAL: // this is a bit of a mess
		if fmtmode == FErr {
			if n.Orig != nil && n.Orig != n {
				return exprfmt(n.Orig, prec)
			}
			if n.Sym != nil {
				return sconv(n.Sym, 0)
			}
		}
		if n.Val().Ctype() == CTNIL && n.Orig != nil && n.Orig != n {
			return exprfmt(n.Orig, prec)
		}
		if n.Type != nil && n.Type.Etype != TIDEAL && n.Type.Etype != TNIL && n.Type != idealbool && n.Type != idealstring {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if n.Type.IsPtr() || (n.Type.IsChan() && n.Type.ChanDir() == Crecv) {
				return fmt.Sprintf("(%v)(%v)", n.Type, vconv(n.Val(), 0))
			} else {
				return fmt.Sprintf("%v(%v)", n.Type, vconv(n.Val(), 0))
			}
		}

		return vconv(n.Val(), 0)

	// Special case: name used as local variable in export.
	// _ becomes ~b%d internally; print as _ for export
	case ONAME:
		if (fmtmode == FExp || fmtmode == FErr) && n.Sym != nil && n.Sym.Name[0] == '~' && n.Sym.Name[1] == 'b' {
			return "_"
		}
		if fmtmode == FExp && n.Sym != nil && !isblank(n) && n.Name.Vargen > 0 {
			return fmt.Sprintf("%v·%d", n.Sym, n.Name.Vargen)
		}

		// Special case: explicit name of func (*T) method(...) is turned into pkg.(*T).method,
		// but for export, this should be rendered as (*pkg.T).meth.
		// These nodes have the special property that they are names with a left OTYPE and a right ONAME.
		if fmtmode == FExp && n.Left != nil && n.Left.Op == OTYPE && n.Right != nil && n.Right.Op == ONAME {
			if n.Left.Type.IsPtr() {
				return fmt.Sprintf("(%v).%v", n.Left.Type, sconv(n.Right.Sym, FmtShort|FmtByte))
			} else {
				return fmt.Sprintf("%v.%v", n.Left.Type, sconv(n.Right.Sym, FmtShort|FmtByte))
			}
		}
		fallthrough

	case OPACK, ONONAME:
		return sconv(n.Sym, 0)

	case OTYPE:
		if n.Type == nil && n.Sym != nil {
			return sconv(n.Sym, 0)
		}
		return Tconv(n.Type, 0)

	case OTARRAY:
		if n.Left != nil {
			return fmt.Sprintf("[]%v", n.Left)
		}
		return fmt.Sprintf("[]%v", n.Right) // happens before typecheck

	case OTMAP:
		return fmt.Sprintf("map[%v]%v", n.Left, n.Right)

	case OTCHAN:
		switch ChanDir(n.Etype) {
		case Crecv:
			return fmt.Sprintf("<-chan %v", n.Left)

		case Csend:
			return fmt.Sprintf("chan<- %v", n.Left)

		default:
			if n.Left != nil && n.Left.Op == OTCHAN && n.Left.Sym == nil && ChanDir(n.Left.Etype) == Crecv {
				return fmt.Sprintf("chan (%v)", n.Left)
			} else {
				return fmt.Sprintf("chan %v", n.Left)
			}
		}

	case OTSTRUCT:
		return "<struct>"

	case OTINTER:
		return "<inter>"

	case OTFUNC:
		return "<func>"

	case OCLOSURE:
		if fmtmode == FErr {
			return "func literal"
		}
		if n.Nbody.Len() != 0 {
			return fmt.Sprintf("%v { %v }", n.Type, n.Nbody)
		}
		return fmt.Sprintf("%v { %v }", n.Type, n.Func.Closure.Nbody)

	case OCOMPLIT:
		ptrlit := n.Right != nil && n.Right.Implicit && n.Right.Type != nil && n.Right.Type.IsPtr()
		if fmtmode == FErr {
			if n.Right != nil && n.Right.Type != nil && !n.Implicit {
				if ptrlit {
					return fmt.Sprintf("&%v literal", n.Right.Type.Elem())
				} else {
					return fmt.Sprintf("%v literal", n.Right.Type)
				}
			}

			return "composite literal"
		}

		if fmtmode == FExp && ptrlit {
			// typecheck has overwritten OIND by OTYPE with pointer type.
			return fmt.Sprintf("(&%v{ %v })", n.Right.Type.Elem(), hconv(n.List, FmtComma))
		}

		return fmt.Sprintf("(%v{ %v })", n.Right, hconv(n.List, FmtComma))

	case OPTRLIT:
		if fmtmode == FExp && n.Left.Implicit {
			return Nconv(n.Left, 0)
		}
		return fmt.Sprintf("&%v", n.Left)

	case OSTRUCTLIT:
		if fmtmode == FExp { // requires special handling of field names
			var f string
			if n.Implicit {
				f += "{"
			} else {
				f += fmt.Sprintf("(%v{", n.Type)
			}
			for i1, n1 := range n.List.Slice() {
				f += fmt.Sprintf(" %v:%v", sconv(n1.Left.Sym, FmtShort|FmtByte), n1.Right)

				if i1+1 < n.List.Len() {
					f += ","
				} else {
					f += " "
				}
			}

			if !n.Implicit {
				f += "})"
				return f
			}
			f += "}"
			return f
		}
		fallthrough

	case OARRAYLIT, OMAPLIT:
		if fmtmode == FErr {
			return fmt.Sprintf("%v literal", n.Type)
		}
		if fmtmode == FExp && n.Implicit {
			return fmt.Sprintf("{ %v }", hconv(n.List, FmtComma))
		}
		return fmt.Sprintf("(%v{ %v })", n.Type, hconv(n.List, FmtComma))

	case OKEY:
		if n.Left != nil && n.Right != nil {
			if fmtmode == FExp && n.Left.Type == structkey {
				// requires special handling of field names
				return fmt.Sprintf("%v:%v", sconv(n.Left.Sym, FmtShort|FmtByte), n.Right)
			} else {
				return fmt.Sprintf("%v:%v", n.Left, n.Right)
			}
		}

		if n.Left == nil && n.Right != nil {
			return fmt.Sprintf(":%v", n.Right)
		}
		if n.Left != nil && n.Right == nil {
			return fmt.Sprintf("%v:", n.Left)
		}
		return ":"

	case OCALLPART:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Right == nil || n.Right.Sym == nil {
			f += ".<nil>"
			return f
		}
		f += fmt.Sprintf(".%v", sconv(n.Right.Sym, FmtShort|FmtByte))
		return f

	case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Sym == nil {
			f += ".<nil>"
			return f
		}
		f += fmt.Sprintf(".%v", sconv(n.Sym, FmtShort|FmtByte))
		return f

	case ODOTTYPE, ODOTTYPE2:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Right != nil {
			f += fmt.Sprintf(".(%v)", n.Right)
			return f
		}
		f += fmt.Sprintf(".(%v)", n.Type)
		return f

	case OINDEX, OINDEXMAP:
		return fmt.Sprintf("%s[%v]", exprfmt(n.Left, nprec), n.Right)

	case OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
		var buf bytes.Buffer
		buf.WriteString(exprfmt(n.Left, nprec))
		buf.WriteString("[")
		low, high, max := n.SliceBounds()
		if low != nil {
			buf.WriteString(low.String())
		}
		buf.WriteString(":")
		if high != nil {
			buf.WriteString(high.String())
		}
		if n.Op.IsSlice3() {
			buf.WriteString(":")
			if max != nil {
				buf.WriteString(max.String())
			}
		}
		buf.WriteString("]")
		return buf.String()

	case OCOPY, OCOMPLEX:
		return fmt.Sprintf("%#v(%v, %v)", n.Op, n.Left, n.Right)

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		ORUNESTR:
		if n.Type == nil || n.Type.Sym == nil {
			return fmt.Sprintf("(%v)(%v)", n.Type, n.Left)
		}
		if n.Left != nil {
			return fmt.Sprintf("%v(%v)", n.Type, n.Left)
		}
		return fmt.Sprintf("%v(%v)", n.Type, hconv(n.List, FmtComma))

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
			return fmt.Sprintf("%#v(%v)", n.Op, n.Left)
		}
		if n.Isddd {
			return fmt.Sprintf("%#v(%v...)", n.Op, hconv(n.List, FmtComma))
		}
		return fmt.Sprintf("%#v(%v)", n.Op, hconv(n.List, FmtComma))

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Isddd {
			f += fmt.Sprintf("(%v...)", hconv(n.List, FmtComma))
			return f
		}
		f += fmt.Sprintf("(%v)", hconv(n.List, FmtComma))
		return f

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		if n.List.Len() != 0 { // pre-typecheck
			return fmt.Sprintf("make(%v, %v)", n.Type, hconv(n.List, FmtComma))
		}
		if n.Right != nil {
			return fmt.Sprintf("make(%v, %v, %v)", n.Type, n.Left, n.Right)
		}
		if n.Left != nil && (n.Op == OMAKESLICE || !n.Left.Type.IsUntyped()) {
			return fmt.Sprintf("make(%v, %v)", n.Type, n.Left)
		}
		return fmt.Sprintf("make(%v)", n.Type)

		// Unary
	case OPLUS,
		OMINUS,
		OADDR,
		OCOM,
		OIND,
		ONOT,
		ORECV:
		f := n.Op.GoString() // %#v
		if n.Left.Op == n.Op {
			f += " "
		}
		f += exprfmt(n.Left, nprec+1)
		return f

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
		var f string
		f += exprfmt(n.Left, nprec)

		f += fmt.Sprintf(" %#v ", n.Op)
		f += exprfmt(n.Right, nprec+1)
		return f

	case OADDSTR:
		var f string
		i := 0
		for _, n1 := range n.List.Slice() {
			if i != 0 {
				f += " + "
			}
			f += exprfmt(n1, nprec)
			i++
		}

		return f

	case OCMPSTR, OCMPIFACE:
		var f string
		f += exprfmt(n.Left, nprec)
		// TODO(marvin): Fix Node.EType type union.
		f += fmt.Sprintf(" %#v ", Op(n.Etype))
		f += exprfmt(n.Right, nprec+1)
		return f

	case ODCLCONST:
		// if exporting, DCLCONST should just be removed as its usage
		// has already been replaced with literals
		if fmtbody {
			return ""
		}
	}

	return fmt.Sprintf("<node %v>", n.Op)
}

func nodefmt(n *Node, flag FmtFlag) string {
	t := n.Type

	// we almost always want the original, except in export mode for literals
	// this saves the importer some work, and avoids us having to redo some
	// special casing for package unsafe
	if (fmtmode != FExp || n.Op != OLITERAL) && n.Orig != nil {
		n = n.Orig
	}

	if flag&FmtLong != 0 && t != nil {
		if t.Etype == TNIL {
			return "nil"
		} else {
			return fmt.Sprintf("%v (type %v)", n, t)
		}
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if opprec[n.Op] < 0 {
		return stmtfmt(n)
	}

	return exprfmt(n, 0)
}

var dumpdepth int

func indent(buf *bytes.Buffer) {
	buf.WriteString("\n")
	for i := 0; i < dumpdepth; i++ {
		buf.WriteString(".   ")
	}
}

func nodedump(n *Node, flag FmtFlag) string {
	if n == nil {
		return ""
	}

	recur := flag&FmtShort == 0

	var buf bytes.Buffer
	if recur {
		indent(&buf)
		if dumpdepth > 10 {
			buf.WriteString("...")
			return buf.String()
		}

		if n.Ninit.Len() != 0 {
			fmt.Fprintf(&buf, "%v-init%v", n.Op, n.Ninit)
			indent(&buf)
		}
	}

	switch n.Op {
	default:
		fmt.Fprintf(&buf, "%v%v", n.Op, jconv(n, 0))

	case OREGISTER, OINDREG:
		fmt.Fprintf(&buf, "%v-%v%v", n.Op, obj.Rconv(int(n.Reg)), jconv(n, 0))

	case OLITERAL:
		fmt.Fprintf(&buf, "%v-%v%v", n.Op, vconv(n.Val(), 0), jconv(n, 0))

	case ONAME, ONONAME:
		if n.Sym != nil {
			fmt.Fprintf(&buf, "%v-%v%v", n.Op, n.Sym, jconv(n, 0))
		} else {
			fmt.Fprintf(&buf, "%v%v", n.Op, jconv(n, 0))
		}
		if recur && n.Type == nil && n.Name != nil && n.Name.Param != nil && n.Name.Param.Ntype != nil {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}

	case OASOP:
		fmt.Fprintf(&buf, "%v-%v%v", n.Op, Op(n.Etype), jconv(n, 0))

	case OTYPE:
		fmt.Fprintf(&buf, "%v %v%v type=%v", n.Op, n.Sym, jconv(n, 0), n.Type)
		if recur && n.Type == nil && n.Name.Param.Ntype != nil {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}
	}

	if n.Sym != nil && n.Op != ONAME {
		fmt.Fprintf(&buf, " %v", n.Sym)
	}

	if n.Type != nil {
		fmt.Fprintf(&buf, " %v", n.Type)
	}

	if recur {
		if n.Left != nil {
			buf.WriteString(Nconv(n.Left, 0))
		}
		if n.Right != nil {
			buf.WriteString(Nconv(n.Right, 0))
		}
		if n.List.Len() != 0 {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-list%v", n.Op, n.List)
		}

		if n.Rlist.Len() != 0 {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-rlist%v", n.Op, n.Rlist)
		}

		if n.Nbody.Len() != 0 {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-body%v", n.Op, n.Nbody)
		}
	}

	return buf.String()
}

func (s *Sym) String() string {
	return sconv(s, 0)
}

// Fmt "%S": syms
// Flags:  "%hS" suppresses qualifying with package
func sconv(s *Sym, flag FmtFlag) string {
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
	str := symfmt(s, flag)
	flag = sf
	fmtmode = sm
	fmtbody = sb
	return str
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
		if (fmtmode == FErr || fmtmode == FExp) && f.Nname != nil {
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
				name = sconv(s, FmtShort|FmtByte) // qualify non-exported names (used on structs, not on funarg)
			} else {
				name = sconv(s, 0)
			}
		} else if fmtmode == FExp {
			if f.Embedded != 0 && s.Pkg != nil && len(s.Pkg.Path) > 0 {
				name = fmt.Sprintf("@%q.?", s.Pkg.Path)
			} else {
				name = "?"
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
	if n == nil {
		return "<N>"
	}
	sf := flag
	sm, sb := setfmode(&flag)

	var str string
	switch fmtmode {
	case FErr, FExp:
		str = nodefmt(n, flag)

	case FDbg:
		dumpdepth++
		str = nodedump(n, flag)
		dumpdepth--

	default:
		Fatalf("unhandled %%N mode")
	}

	flag = sf
	fmtbody = sb
	fmtmode = sm
	return str
}

func (n Nodes) String() string {
	return hconv(n, 0)
}

// Fmt '%H': Nodes.
// Flags: all those of %N plus ',': separate with comma's instead of semicolons.
func hconv(l Nodes, flag FmtFlag) string {
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

	var buf bytes.Buffer
	for i, n := range l.Slice() {
		buf.WriteString(Nconv(n, 0))
		if i+1 < l.Len() {
			buf.WriteString(sep)
		}
	}

	flag = sf
	fmtbody = sb
	fmtmode = sm
	return buf.String()
}

func dumplist(s string, l Nodes) {
	fmt.Printf("%s%v\n", s, hconv(l, FmtSign))
}

func Dump(s string, n *Node) {
	fmt.Printf("%s [%p]%v\n", s, n, Nconv(n, FmtSign))
}
