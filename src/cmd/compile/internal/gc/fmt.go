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
//	%H NodeList*	NodeLists
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
// recursions of %N, %T and %S, but not the h and l flags.  The u flag is
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

func setfmode(flags *int) (fm int, fb bool) {
	fm = fmtmode
	fb = fmtbody
	if *flags&obj.FmtSign != 0 {
		fmtmode = FDbg
	} else if *flags&obj.FmtSharp != 0 {
		fmtmode = FExp
	} else if *flags&obj.FmtLeft != 0 {
		fmtmode = FTypeId
	}

	if *flags&obj.FmtBody != 0 {
		fmtbody = true
	}

	*flags &^= (obj.FmtSharp | obj.FmtLeft | obj.FmtSign | obj.FmtBody)
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
}

// Fmt "%O":  Node opcodes
func Oconv(o int, flag int) string {
	if (flag&obj.FmtSharp != 0) || fmtmode != FDbg {
		if o >= 0 && o < len(goopnames) && goopnames[o] != "" {
			return goopnames[o]
		}
	}

	if o >= 0 && o < len(opnames) && opnames[o] != "" {
		return opnames[o]
	}

	return fmt.Sprintf("O-%d", o)
}

var classnames = []string{
	"Pxxx",
	"PEXTERN",
	"PAUTO",
	"PPARAM",
	"PPARAMOUT",
	"PPARAMREF",
	"PFUNC",
}

// Fmt "%J": Node details.
func Jconv(n *Node, flag int) string {
	var buf bytes.Buffer

	c := flag & obj.FmtShort

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
		s := ""
		if n.Class&PHEAP != 0 {
			s = ",heap"
		}
		if int(n.Class&^PHEAP) < len(classnames) {
			fmt.Fprintf(&buf, " class(%s%s)", classnames[n.Class&^PHEAP], s)
		} else {
			fmt.Fprintf(&buf, " class(%d?%s)", n.Class&^PHEAP, s)
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

	if c == 0 && n.Used {
		fmt.Fprintf(&buf, " used(%v)", n.Used)
	}
	return buf.String()
}

// Fmt "%V": Values
func Vconv(v Val, flag int) string {
	switch v.Ctype() {
	case CTINT:
		if (flag&obj.FmtSharp != 0) || fmtmode == FExp {
			return Bconv(v.U.(*Mpint), obj.FmtSharp)
		}
		return Bconv(v.U.(*Mpint), 0)

	case CTRUNE:
		x := Mpgetfix(v.U.(*Mpint))
		if ' ' <= x && x < 0x80 && x != '\\' && x != '\'' {
			return fmt.Sprintf("'%c'", int(x))
		}
		if 0 <= x && x < 1<<16 {
			return fmt.Sprintf("'\\u%04x'", uint(int(x)))
		}
		if 0 <= x && x <= utf8.MaxRune {
			return fmt.Sprintf("'\\U%08x'", uint64(x))
		}
		return fmt.Sprintf("('\\x00' + %v)", v.U.(*Mpint))

	case CTFLT:
		if (flag&obj.FmtSharp != 0) || fmtmode == FExp {
			return Fconv(v.U.(*Mpflt), 0)
		}
		return Fconv(v.U.(*Mpflt), obj.FmtSharp)

	case CTCPLX:
		if (flag&obj.FmtSharp != 0) || fmtmode == FExp {
			return fmt.Sprintf("(%v+%vi)", &v.U.(*Mpcplx).Real, &v.U.(*Mpcplx).Imag)
		}
		if mpcmpfltc(&v.U.(*Mpcplx).Real, 0) == 0 {
			return fmt.Sprintf("%vi", Fconv(&v.U.(*Mpcplx).Imag, obj.FmtSharp))
		}
		if mpcmpfltc(&v.U.(*Mpcplx).Imag, 0) == 0 {
			return Fconv(&v.U.(*Mpcplx).Real, obj.FmtSharp)
		}
		if mpcmpfltc(&v.U.(*Mpcplx).Imag, 0) < 0 {
			return fmt.Sprintf("(%v%vi)", Fconv(&v.U.(*Mpcplx).Real, obj.FmtSharp), Fconv(&v.U.(*Mpcplx).Imag, obj.FmtSharp))
		}
		return fmt.Sprintf("(%v+%vi)", Fconv(&v.U.(*Mpcplx).Real, obj.FmtSharp), Fconv(&v.U.(*Mpcplx).Imag, obj.FmtSharp))

	case CTSTR:
		return strconv.Quote(v.U.(string))

	case CTBOOL:
		if v.U.(bool) {
			return "true"
		}
		return "false"

	case CTNIL:
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
	TSTRUCT:     "STRUCT",
	TCHAN:       "CHAN",
	TMAP:        "MAP",
	TINTER:      "INTER",
	TFORW:       "FORW",
	TFIELD:      "FIELD",
	TSTRING:     "STRING",
	TANY:        "ANY",
}

// Fmt "%E": etype
func Econv(et EType) string {
	if int(et) < len(etnames) && etnames[et] != "" {
		return etnames[et]
	}
	return fmt.Sprintf("E-%d", et)
}

// Fmt "%S": syms
func symfmt(s *Sym, flag int) string {
	if s.Pkg != nil && flag&obj.FmtShort == 0 {
		switch fmtmode {
		case FErr: // This is for the user
			if s.Pkg == localpkg {
				return s.Name
			}

			// If the name was used by multiple packages, display the full path,
			if s.Pkg.Name != "" && numImport[s.Pkg.Name] > 1 {
				return fmt.Sprintf("%q.%s", s.Pkg.Path, s.Name)
			}
			return fmt.Sprintf("%s.%s", s.Pkg.Name, s.Name)

		case FDbg:
			return fmt.Sprintf("%s.%s", s.Pkg.Name, s.Name)

		case FTypeId:
			if flag&obj.FmtUnsigned != 0 {
				return fmt.Sprintf("%s.%s", s.Pkg.Name, s.Name) // dcommontype, typehash
			}
			return fmt.Sprintf("%s.%s", s.Pkg.Prefix, s.Name) // (methodsym), typesym, weaksym

		case FExp:
			if s.Name != "" && s.Name[0] == '.' {
				Fatalf("exporting synthetic symbol %s", s.Name)
			}
			if s.Pkg != builtinpkg {
				return fmt.Sprintf("@%q.%s", s.Pkg.Path, s.Name)
			}
		}
	}

	if flag&obj.FmtByte != 0 {
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

func typefmt(t *Type, flag int) string {
	if t == nil {
		return "<T>"
	}

	if t == bytetype || t == runetype {
		// in %-T mode collapse rune and byte with their originals.
		if fmtmode != FTypeId {
			return Sconv(t.Sym, obj.FmtShort)
		}
		t = Types[t.Etype]
	}

	if t == errortype {
		return "error"
	}

	// Unless the 'l' flag was specified, if the type has a name, just print that name.
	if flag&obj.FmtLong == 0 && t.Sym != nil && t.Etype != TFIELD && t != Types[t.Etype] {
		switch fmtmode {
		case FTypeId:
			if flag&obj.FmtShort != 0 {
				if t.Vargen != 0 {
					return fmt.Sprintf("%v·%d", Sconv(t.Sym, obj.FmtShort), t.Vargen)
				}
				return Sconv(t.Sym, obj.FmtShort)
			}

			if flag&obj.FmtUnsigned != 0 {
				return Sconv(t.Sym, obj.FmtUnsigned)
			}
			fallthrough

		case FExp:
			if t.Sym.Pkg == localpkg && t.Vargen != 0 {
				return fmt.Sprintf("%v·%d", t.Sym, t.Vargen)
			}
		}

		return Sconv(t.Sym, 0)
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
		str := Econv(t.Etype) + "-" + typefmt(t, flag)
		fmtmode = FDbg
		return str
	}

	switch t.Etype {
	case TPTR32, TPTR64:
		if fmtmode == FTypeId && (flag&obj.FmtShort != 0) {
			return fmt.Sprintf("*%v", Tconv(t.Type, obj.FmtShort))
		}
		return fmt.Sprintf("*%v", t.Type)

	case TARRAY:
		if t.Bound >= 0 {
			return fmt.Sprintf("[%d]%v", t.Bound, t.Type)
		}
		if t.Bound == -100 {
			return fmt.Sprintf("[...]%v", t.Type)
		}
		return fmt.Sprintf("[]%v", t.Type)

	case TCHAN:
		switch t.Chan {
		case Crecv:
			return fmt.Sprintf("<-chan %v", t.Type)

		case Csend:
			return fmt.Sprintf("chan<- %v", t.Type)
		}

		if t.Type != nil && t.Type.Etype == TCHAN && t.Type.Sym == nil && t.Type.Chan == Crecv {
			return fmt.Sprintf("chan (%v)", t.Type)
		}
		return fmt.Sprintf("chan %v", t.Type)

	case TMAP:
		return fmt.Sprintf("map[%v]%v", t.Down, t.Type)

	case TINTER:
		var buf bytes.Buffer
		buf.WriteString("interface {")
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			buf.WriteString(" ")
			switch {
			case t1.Sym == nil:
				// Check first that a symbol is defined for this type.
				// Wrong interface definitions may have types lacking a symbol.
				break
			case exportname(t1.Sym.Name):
				buf.WriteString(Sconv(t1.Sym, obj.FmtShort))
			default:
				buf.WriteString(Sconv(t1.Sym, obj.FmtUnsigned))
			}
			buf.WriteString(Tconv(t1.Type, obj.FmtShort))
			if t1.Down != nil {
				buf.WriteString(";")
			}
		}
		if t.Type != nil {
			buf.WriteString(" ")
		}
		buf.WriteString("}")
		return buf.String()

	case TFUNC:
		var buf bytes.Buffer
		if flag&obj.FmtShort != 0 {
			// no leading func
		} else {
			if t.Thistuple != 0 {
				buf.WriteString("method")
				buf.WriteString(Tconv(getthisx(t), 0))
				buf.WriteString(" ")
			}
			buf.WriteString("func")
		}
		buf.WriteString(Tconv(getinargx(t), 0))

		switch t.Outtuple {
		case 0:
			break

		case 1:
			if fmtmode != FExp {
				buf.WriteString(" ")
				buf.WriteString(Tconv(getoutargx(t).Type.Type, 0)) // struct->field->field's type
				break
			}
			fallthrough

		default:
			buf.WriteString(" ")
			buf.WriteString(Tconv(getoutargx(t), 0))
		}
		return buf.String()

	case TSTRUCT:
		if t.Map != nil {
			// Format the bucket struct for map[x]y as map.bucket[x]y.
			// This avoids a recursive print that generates very long names.
			if t.Map.Bucket == t {
				return fmt.Sprintf("map.bucket[%v]%v", t.Map.Down, t.Map.Type)
			}

			if t.Map.Hmap == t {
				return fmt.Sprintf("map.hdr[%v]%v", t.Map.Down, t.Map.Type)
			}

			if t.Map.Hiter == t {
				return fmt.Sprintf("map.iter[%v]%v", t.Map.Down, t.Map.Type)
			}

			Yyerror("unknown internal map type")
		}

		var buf bytes.Buffer
		if t.Funarg {
			buf.WriteString("(")
			if fmtmode == FTypeId || fmtmode == FErr { // no argument names on function signature, and no "noescape"/"nosplit" tags
				for t1 := t.Type; t1 != nil; t1 = t1.Down {
					buf.WriteString(Tconv(t1, obj.FmtShort))
					if t1.Down != nil {
						buf.WriteString(", ")
					}
				}
			} else {
				for t1 := t.Type; t1 != nil; t1 = t1.Down {
					buf.WriteString(Tconv(t1, 0))
					if t1.Down != nil {
						buf.WriteString(", ")
					}
				}
			}
			buf.WriteString(")")
		} else {
			buf.WriteString("struct {")
			for t1 := t.Type; t1 != nil; t1 = t1.Down {
				buf.WriteString(" ")
				buf.WriteString(Tconv(t1, obj.FmtLong))
				if t1.Down != nil {
					buf.WriteString(";")
				}
			}
			if t.Type != nil {
				buf.WriteString(" ")
			}
			buf.WriteString("}")
		}
		return buf.String()

	case TFIELD:
		var name string
		if flag&obj.FmtShort == 0 {
			s := t.Sym

			// Take the name from the original, lest we substituted it with ~r%d or ~b%d.
			// ~r%d is a (formerly) unnamed result.
			if (fmtmode == FErr || fmtmode == FExp) && t.Nname != nil {
				if t.Nname.Orig != nil {
					s = t.Nname.Orig.Sym
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

			if s != nil && t.Embedded == 0 {
				if t.Funarg {
					name = Nconv(t.Nname, 0)
				} else if flag&obj.FmtLong != 0 {
					name = Sconv(s, obj.FmtShort|obj.FmtByte) // qualify non-exported names (used on structs, not on funarg)
				} else {
					name = Sconv(s, 0)
				}
			} else if fmtmode == FExp {
				// TODO(rsc) this breaks on the eliding of unused arguments in the backend
				// when this is fixed, the special case in dcl.go checkarglist can go.
				//if(t->funarg)
				//	fmtstrcpy(fp, "_ ");
				//else
				if t.Embedded != 0 && s.Pkg != nil && len(s.Pkg.Path) > 0 {
					name = fmt.Sprintf("@%q.?", s.Pkg.Path)
				} else {
					name = "?"
				}
			}
		}

		var typ string
		if t.Isddd {
			typ = "..." + Tconv(t.Type.Type, 0)
		} else {
			typ = Tconv(t.Type, 0)
		}

		str := typ
		if name != "" {
			str = name + " " + typ
		}
		if flag&obj.FmtShort == 0 && !fmtbody && t.Note != nil {
			str += " " + strconv.Quote(*t.Note)
		}
		return str

	case TFORW:
		if t.Sym != nil {
			return fmt.Sprintf("undefined %v", t.Sym)
		}
		return "undefined"

	case TUNSAFEPTR:
		if fmtmode == FExp {
			return "@\"unsafe\".Pointer"
		}
		return "unsafe.Pointer"
	}

	if fmtmode == FExp {
		Fatalf("missing %v case during export", Econv(t.Etype))
	}

	// Don't know how to handle - fall back to detailed prints.
	return fmt.Sprintf("%v <%v> %v", Econv(t.Etype), t.Sym, t.Type)
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
	// and inlining.  If it doesn't fit the syntax, emit an enclosing
	// block starting with the init statements.

	// if we can just say "for" n->ninit; ... then do so
	simpleinit := n.Ninit != nil && n.Ninit.Next == nil && n.Ninit.N.Ninit == nil && stmtwithinit(n.Op)

	// otherwise, print the inits as separate statements
	complexinit := n.Ninit != nil && !simpleinit && (fmtmode != FErr)

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
			switch n.Left.Class &^ PHEAP {
			case PPARAM, PPARAMOUT, PAUTO:
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
	// preceded by the DCL which will be re-parsed and typecheck to reproduce
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

		f += fmt.Sprintf("%v %v= %v", n.Left, Oconv(int(n.Etype), obj.FmtSharp), n.Right)

	case OAS2:
		if n.Colas && !complexinit {
			f += fmt.Sprintf("%v := %v", Hconv(n.List, obj.FmtComma), Hconv(n.Rlist, obj.FmtComma))
			break
		}
		fallthrough

	case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		f += fmt.Sprintf("%v = %v", Hconv(n.List, obj.FmtComma), Hconv(n.Rlist, obj.FmtComma))

	case ORETURN:
		f += fmt.Sprintf("return %v", Hconv(n.List, obj.FmtComma))

	case ORETJMP:
		f += fmt.Sprintf("retjmp %v", n.Sym)

	case OPROC:
		f += fmt.Sprintf("go %v", n.Left)

	case ODEFER:
		f += fmt.Sprintf("defer %v", n.Left)

	case OIF:
		if simpleinit {
			f += fmt.Sprintf("if %v; %v { %v }", n.Ninit.N, n.Left, n.Nbody)
		} else {
			f += fmt.Sprintf("if %v { %v }", n.Left, n.Nbody)
		}
		if n.Rlist != nil {
			f += fmt.Sprintf(" else { %v }", n.Rlist)
		}

	case OFOR:
		if fmtmode == FErr { // TODO maybe only if FmtShort, same below
			f += "for loop"
			break
		}

		f += "for"
		if simpleinit {
			f += fmt.Sprintf(" %v;", n.Ninit.N)
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

		if n.List == nil {
			f += fmt.Sprintf("for range %v { %v }", n.Right, n.Nbody)
			break
		}

		f += fmt.Sprintf("for %v = range %v { %v }", Hconv(n.List, obj.FmtComma), n.Right, n.Nbody)

	case OSELECT, OSWITCH:
		if fmtmode == FErr {
			f += fmt.Sprintf("%v statement", Oconv(int(n.Op), 0))
			break
		}

		f += Oconv(int(n.Op), obj.FmtSharp)
		if simpleinit {
			f += fmt.Sprintf(" %v;", n.Ninit.N)
		}
		if n.Left != nil {
			f += Nconv(n.Left, 0)
		}

		f += fmt.Sprintf(" { %v }", n.List)

	case OCASE, OXCASE:
		if n.List != nil {
			f += fmt.Sprintf("case %v: %v", Hconv(n.List, obj.FmtComma), n.Nbody)
		} else {
			f += fmt.Sprintf("default: %v", n.Nbody)
		}

	case OBREAK,
		OCONTINUE,
		OGOTO,
		OFALL,
		OXFALL:
		if n.Left != nil {
			f += fmt.Sprintf("%v %v", Oconv(int(n.Op), obj.FmtSharp), n.Left)
		} else {
			f += Oconv(int(n.Op), obj.FmtSharp)
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
				return Sconv(n.Sym, 0)
			}
		}
		if n.Val().Ctype() == CTNIL && n.Orig != nil && n.Orig != n {
			return exprfmt(n.Orig, prec)
		}
		if n.Type != nil && n.Type.Etype != TIDEAL && n.Type.Etype != TNIL && n.Type != idealbool && n.Type != idealstring {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if Isptr[n.Type.Etype] || (n.Type.Etype == TCHAN && n.Type.Chan == Crecv) {
				return fmt.Sprintf("(%v)(%v)", n.Type, Vconv(n.Val(), 0))
			} else {
				return fmt.Sprintf("%v(%v)", n.Type, Vconv(n.Val(), 0))
			}
		}

		return Vconv(n.Val(), 0)

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
			if Isptr[n.Left.Type.Etype] {
				return fmt.Sprintf("(%v).%v", n.Left.Type, Sconv(n.Right.Sym, obj.FmtShort|obj.FmtByte))
			} else {
				return fmt.Sprintf("%v.%v", n.Left.Type, Sconv(n.Right.Sym, obj.FmtShort|obj.FmtByte))
			}
		}
		fallthrough

	case OPACK, ONONAME:
		return Sconv(n.Sym, 0)

	case OTYPE:
		if n.Type == nil && n.Sym != nil {
			return Sconv(n.Sym, 0)
		}
		return Tconv(n.Type, 0)

	case OTARRAY:
		if n.Left != nil {
			return fmt.Sprintf("[]%v", n.Left)
		}
		var f string
		f += fmt.Sprintf("[]%v", n.Right)
		return f // happens before typecheck

	case OTMAP:
		return fmt.Sprintf("map[%v]%v", n.Left, n.Right)

	case OTCHAN:
		switch n.Etype {
		case Crecv:
			return fmt.Sprintf("<-chan %v", n.Left)

		case Csend:
			return fmt.Sprintf("chan<- %v", n.Left)

		default:
			if n.Left != nil && n.Left.Op == OTCHAN && n.Left.Sym == nil && n.Left.Etype == Crecv {
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
		if n.Nbody != nil {
			return fmt.Sprintf("%v { %v }", n.Type, n.Nbody)
		}
		return fmt.Sprintf("%v { %v }", n.Type, n.Name.Param.Closure.Nbody)

	case OCOMPLIT:
		ptrlit := n.Right != nil && n.Right.Implicit && n.Right.Type != nil && Isptr[n.Right.Type.Etype]
		if fmtmode == FErr {
			if n.Right != nil && n.Right.Type != nil && !n.Implicit {
				if ptrlit {
					return fmt.Sprintf("&%v literal", n.Right.Type.Type)
				} else {
					return fmt.Sprintf("%v literal", n.Right.Type)
				}
			}

			return "composite literal"
		}

		if fmtmode == FExp && ptrlit {
			// typecheck has overwritten OIND by OTYPE with pointer type.
			return fmt.Sprintf("(&%v{ %v })", n.Right.Type.Type, Hconv(n.List, obj.FmtComma))
		}

		return fmt.Sprintf("(%v{ %v })", n.Right, Hconv(n.List, obj.FmtComma))

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
			for l := n.List; l != nil; l = l.Next {
				f += fmt.Sprintf(" %v:%v", Sconv(l.N.Left.Sym, obj.FmtShort|obj.FmtByte), l.N.Right)

				if l.Next != nil {
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
			return fmt.Sprintf("{ %v }", Hconv(n.List, obj.FmtComma))
		}
		return fmt.Sprintf("(%v{ %v })", n.Type, Hconv(n.List, obj.FmtComma))

	case OKEY:
		if n.Left != nil && n.Right != nil {
			if fmtmode == FExp && n.Left.Type != nil && n.Left.Type.Etype == TFIELD {
				// requires special handling of field names
				return fmt.Sprintf("%v:%v", Sconv(n.Left.Sym, obj.FmtShort|obj.FmtByte), n.Right)
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

	case OXDOT,
		ODOT,
		ODOTPTR,
		ODOTINTER,
		ODOTMETH,
		OCALLPART:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Right == nil || n.Right.Sym == nil {
			f += ".<nil>"
			return f
		}
		f += fmt.Sprintf(".%v", Sconv(n.Right.Sym, obj.FmtShort|obj.FmtByte))
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

	case OINDEX,
		OINDEXMAP,
		OSLICE,
		OSLICESTR,
		OSLICEARR,
		OSLICE3,
		OSLICE3ARR:
		var f string
		f += exprfmt(n.Left, nprec)
		f += fmt.Sprintf("[%v]", n.Right)
		return f

	case OCOPY, OCOMPLEX:
		return fmt.Sprintf("%v(%v, %v)", Oconv(int(n.Op), obj.FmtSharp), n.Left, n.Right)

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
		return fmt.Sprintf("%v(%v)", n.Type, Hconv(n.List, obj.FmtComma))

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
			return fmt.Sprintf("%v(%v)", Oconv(int(n.Op), obj.FmtSharp), n.Left)
		}
		if n.Isddd {
			return fmt.Sprintf("%v(%v...)", Oconv(int(n.Op), obj.FmtSharp), Hconv(n.List, obj.FmtComma))
		}
		return fmt.Sprintf("%v(%v)", Oconv(int(n.Op), obj.FmtSharp), Hconv(n.List, obj.FmtComma))

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Isddd {
			f += fmt.Sprintf("(%v...)", Hconv(n.List, obj.FmtComma))
			return f
		}
		f += fmt.Sprintf("(%v)", Hconv(n.List, obj.FmtComma))
		return f

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		if n.List != nil { // pre-typecheck
			return fmt.Sprintf("make(%v, %v)", n.Type, Hconv(n.List, obj.FmtComma))
		}
		if n.Right != nil {
			return fmt.Sprintf("make(%v, %v, %v)", n.Type, n.Left, n.Right)
		}
		if n.Left != nil && (n.Op == OMAKESLICE || !isideal(n.Left.Type)) {
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
		var f string
		if n.Left.Op == n.Op {
			f += fmt.Sprintf("%v ", Oconv(int(n.Op), obj.FmtSharp))
		} else {
			f += Oconv(int(n.Op), obj.FmtSharp)
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

		f += fmt.Sprintf(" %v ", Oconv(int(n.Op), obj.FmtSharp))
		f += exprfmt(n.Right, nprec+1)
		return f

	case OADDSTR:
		var f string
		for l := n.List; l != nil; l = l.Next {
			if l != n.List {
				f += " + "
			}
			f += exprfmt(l.N, nprec)
		}

		return f

	case OCMPSTR, OCMPIFACE:
		var f string
		f += exprfmt(n.Left, nprec)
		// TODO(marvin): Fix Node.EType type union.
		f += fmt.Sprintf(" %v ", Oconv(int(n.Etype), obj.FmtSharp))
		f += exprfmt(n.Right, nprec+1)
		return f
	}

	return fmt.Sprintf("<node %v>", Oconv(int(n.Op), 0))
}

func nodefmt(n *Node, flag int) string {
	t := n.Type

	// we almost always want the original, except in export mode for literals
	// this saves the importer some work, and avoids us having to redo some
	// special casing for package unsafe
	if (fmtmode != FExp || n.Op != OLITERAL) && n.Orig != nil {
		n = n.Orig
	}

	if flag&obj.FmtLong != 0 && t != nil {
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

func nodedump(n *Node, flag int) string {
	if n == nil {
		return ""
	}

	recur := flag&obj.FmtShort == 0

	var buf bytes.Buffer
	if recur {
		indent(&buf)
		if dumpdepth > 10 {
			buf.WriteString("...")
			return buf.String()
		}

		if n.Ninit != nil {
			fmt.Fprintf(&buf, "%v-init%v", Oconv(int(n.Op), 0), n.Ninit)
			indent(&buf)
		}
	}

	switch n.Op {
	default:
		fmt.Fprintf(&buf, "%v%v", Oconv(int(n.Op), 0), Jconv(n, 0))

	case OREGISTER, OINDREG:
		fmt.Fprintf(&buf, "%v-%v%v", Oconv(int(n.Op), 0), obj.Rconv(int(n.Reg)), Jconv(n, 0))

	case OLITERAL:
		fmt.Fprintf(&buf, "%v-%v%v", Oconv(int(n.Op), 0), Vconv(n.Val(), 0), Jconv(n, 0))

	case ONAME, ONONAME:
		if n.Sym != nil {
			fmt.Fprintf(&buf, "%v-%v%v", Oconv(int(n.Op), 0), n.Sym, Jconv(n, 0))
		} else {
			fmt.Fprintf(&buf, "%v%v", Oconv(int(n.Op), 0), Jconv(n, 0))
		}
		if recur && n.Type == nil && n.Name.Param.Ntype != nil {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-ntype%v", Oconv(int(n.Op), 0), n.Name.Param.Ntype)
		}

	case OASOP:
		fmt.Fprintf(&buf, "%v-%v%v", Oconv(int(n.Op), 0), Oconv(int(n.Etype), 0), Jconv(n, 0))

	case OTYPE:
		fmt.Fprintf(&buf, "%v %v%v type=%v", Oconv(int(n.Op), 0), n.Sym, Jconv(n, 0), n.Type)
		if recur && n.Type == nil && n.Name.Param.Ntype != nil {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-ntype%v", Oconv(int(n.Op), 0), n.Name.Param.Ntype)
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
		if n.List != nil {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-list%v", Oconv(int(n.Op), 0), n.List)
		}

		if n.Rlist != nil {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-rlist%v", Oconv(int(n.Op), 0), n.Rlist)
		}

		if n.Nbody != nil {
			indent(&buf)
			fmt.Fprintf(&buf, "%v-body%v", Oconv(int(n.Op), 0), n.Nbody)
		}
	}

	return buf.String()
}

func (s *Sym) String() string {
	return Sconv(s, 0)
}

// Fmt "%S": syms
// Flags:  "%hS" suppresses qualifying with package
func Sconv(s *Sym, flag int) string {
	if flag&obj.FmtLong != 0 {
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

// Fmt "%T": types.
// Flags: 'l' print definition, not name
//	  'h' omit 'func' and receiver from function types, short type names
//	  'u' package name, not prefix (FTypeId mode, sticky)
func Tconv(t *Type, flag int) string {
	if t == nil {
		return "<T>"
	}

	if t.Trecur > 4 {
		return "<...>"
	}

	t.Trecur++
	sf := flag
	sm, sb := setfmode(&flag)

	if fmtmode == FTypeId && (sf&obj.FmtUnsigned != 0) {
		fmtpkgpfx++
	}
	if fmtpkgpfx != 0 {
		flag |= obj.FmtUnsigned
	}

	str := typefmt(t, flag)

	if fmtmode == FTypeId && (sf&obj.FmtUnsigned != 0) {
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
func Nconv(n *Node, flag int) string {
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

func (l *NodeList) String() string {
	return Hconv(l, 0)
}

// Fmt '%H': NodeList.
// Flags: all those of %N plus ',': separate with comma's instead of semicolons.
func Hconv(l *NodeList, flag int) string {
	if l == nil && fmtmode == FDbg {
		return "<nil>"
	}

	sf := flag
	sm, sb := setfmode(&flag)
	sep := "; "
	if fmtmode == FDbg {
		sep = "\n"
	} else if flag&obj.FmtComma != 0 {
		sep = ", "
	}

	var buf bytes.Buffer
	for ; l != nil; l = l.Next {
		buf.WriteString(Nconv(l.N, 0))
		if l.Next != nil {
			buf.WriteString(sep)
		}
	}

	flag = sf
	fmtbody = sb
	fmtmode = sm
	return buf.String()
}

func dumplist(s string, l *NodeList) {
	fmt.Printf("%s%v\n", s, Hconv(l, obj.FmtSign))
}

func Dump(s string, n *Node) {
	fmt.Printf("%s [%p]%v\n", s, n, Nconv(n, obj.FmtSign))
}
