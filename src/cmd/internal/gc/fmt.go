// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
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
//	%Z Strlit*	String literals
//
//   In mparith1.c:
//      %B Mpint*	Big integers
//	%F Mpflt*	Big floats
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

func setfmode(flags *int) int {
	fm := fmtmode
	if *flags&obj.FmtSign != 0 {
		fmtmode = FDbg
	} else if *flags&obj.FmtSharp != 0 {
		fmtmode = FExp
	} else if *flags&obj.FmtLeft != 0 {
		fmtmode = FTypeId
	}

	*flags &^= (obj.FmtSharp | obj.FmtLeft | obj.FmtSign)
	return fm
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
	if (flag&obj.FmtSharp != 0 /*untyped*/) || fmtmode != FDbg {
		if o >= 0 && o < len(goopnames) && goopnames[o] != "" {
			var fp string
			fp += goopnames[o]
			return fp
		}
	}

	if o >= 0 && o < len(opnames) && opnames[o] != "" {
		var fp string
		fp += opnames[o]
		return fp
	}

	var fp string
	fp += fmt.Sprintf("O-%d", o)
	return fp
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
	var fp string

	c := flag & obj.FmtShort

	if c == 0 && n.Ullman != 0 {
		fp += fmt.Sprintf(" u(%d)", n.Ullman)
	}

	if c == 0 && n.Addable != 0 {
		fp += fmt.Sprintf(" a(%d)", n.Addable)
	}

	if c == 0 && n.Vargen != 0 {
		fp += fmt.Sprintf(" g(%d)", n.Vargen)
	}

	if n.Lineno != 0 {
		fp += fmt.Sprintf(" l(%d)", n.Lineno)
	}

	if c == 0 && n.Xoffset != BADWIDTH {
		fp += fmt.Sprintf(" x(%d%+d)", n.Xoffset, n.Stkdelta)
	}

	if n.Class != 0 {
		s := ""
		if n.Class&PHEAP != 0 {
			s = ",heap"
		}
		if int(n.Class&^PHEAP) < len(classnames) {
			fp += fmt.Sprintf(" class(%s%s)", classnames[n.Class&^PHEAP], s)
		} else {
			fp += fmt.Sprintf(" class(%d?%s)", n.Class&^PHEAP, s)
		}
	}

	if n.Colas != 0 {
		fp += fmt.Sprintf(" colas(%d)", n.Colas)
	}

	if n.Funcdepth != 0 {
		fp += fmt.Sprintf(" f(%d)", n.Funcdepth)
	}

	switch n.Esc {
	case EscUnknown:
		break

	case EscHeap:
		fp += fmt.Sprintf(" esc(h)")

	case EscScope:
		fp += fmt.Sprintf(" esc(s)")

	case EscNone:
		fp += fmt.Sprintf(" esc(no)")

	case EscNever:
		if c == 0 {
			fp += fmt.Sprintf(" esc(N)")
		}

	default:
		fp += fmt.Sprintf(" esc(%d)", n.Esc)
	}

	if n.Escloopdepth != 0 {
		fp += fmt.Sprintf(" ld(%d)", n.Escloopdepth)
	}

	if c == 0 && n.Typecheck != 0 {
		fp += fmt.Sprintf(" tc(%d)", n.Typecheck)
	}

	if c == 0 && n.Dodata != 0 {
		fp += fmt.Sprintf(" dd(%d)", n.Dodata)
	}

	if n.Isddd != 0 {
		fp += fmt.Sprintf(" isddd(%d)", n.Isddd)
	}

	if n.Implicit != 0 {
		fp += fmt.Sprintf(" implicit(%d)", n.Implicit)
	}

	if n.Embedded != 0 {
		fp += fmt.Sprintf(" embedded(%d)", n.Embedded)
	}

	if n.Addrtaken != 0 {
		fp += fmt.Sprintf(" addrtaken")
	}

	if n.Assigned != 0 {
		fp += fmt.Sprintf(" assigned")
	}

	if c == 0 && n.Used != 0 {
		fp += fmt.Sprintf(" used(%d)", n.Used)
	}
	return fp
}

// Fmt "%V": Values
func Vconv(v *Val, flag int) string {
	switch v.Ctype {
	case CTINT:
		if (flag&obj.FmtSharp != 0 /*untyped*/) || fmtmode == FExp {
			var fp string
			fp += fmt.Sprintf("%v", Bconv(v.U.Xval, obj.FmtSharp))
			return fp
		}
		var fp string
		fp += fmt.Sprintf("%v", Bconv(v.U.Xval, 0))
		return fp

	case CTRUNE:
		x := Mpgetfix(v.U.Xval)
		if ' ' <= x && x < 0x80 && x != '\\' && x != '\'' {
			var fp string
			fp += fmt.Sprintf("'%c'", int(x))
			return fp
		}
		if 0 <= x && x < 1<<16 {
			var fp string
			fp += fmt.Sprintf("'\\u%04x'", uint(int(x)))
			return fp
		}
		if 0 <= x && x <= utf8.MaxRune {
			var fp string
			fp += fmt.Sprintf("'\\U%08x'", uint64(x))
			return fp
		}
		var fp string
		fp += fmt.Sprintf("('\\x00' + %v)", Bconv(v.U.Xval, 0))
		return fp

	case CTFLT:
		if (flag&obj.FmtSharp != 0 /*untyped*/) || fmtmode == FExp {
			var fp string
			fp += fmt.Sprintf("%v", Fconv(v.U.Fval, 0))
			return fp
		}
		var fp string
		fp += fmt.Sprintf("%v", Fconv(v.U.Fval, obj.FmtSharp))
		return fp

	case CTCPLX:
		if (flag&obj.FmtSharp != 0 /*untyped*/) || fmtmode == FExp {
			var fp string
			fp += fmt.Sprintf("(%v+%vi)", Fconv(&v.U.Cval.Real, 0), Fconv(&v.U.Cval.Imag, 0))
			return fp
		}
		if mpcmpfltc(&v.U.Cval.Real, 0) == 0 {
			var fp string
			fp += fmt.Sprintf("%vi", Fconv(&v.U.Cval.Imag, obj.FmtSharp))
			return fp
		}
		if mpcmpfltc(&v.U.Cval.Imag, 0) == 0 {
			var fp string
			fp += fmt.Sprintf("%v", Fconv(&v.U.Cval.Real, obj.FmtSharp))
			return fp
		}
		if mpcmpfltc(&v.U.Cval.Imag, 0) < 0 {
			var fp string
			fp += fmt.Sprintf("(%v%vi)", Fconv(&v.U.Cval.Real, obj.FmtSharp), Fconv(&v.U.Cval.Imag, obj.FmtSharp))
			return fp
		}
		var fp string
		fp += fmt.Sprintf("(%v+%vi)", Fconv(&v.U.Cval.Real, obj.FmtSharp), Fconv(&v.U.Cval.Imag, obj.FmtSharp))
		return fp

	case CTSTR:
		var fp string
		fp += fmt.Sprintf("\"%v\"", Zconv(v.U.Sval, 0))
		return fp

	case CTBOOL:
		if v.U.Bval != 0 {
			var fp string
			fp += "true"
			return fp
		}
		var fp string
		fp += "false"
		return fp

	case CTNIL:
		var fp string
		fp += "nil"
		return fp
	}

	var fp string
	fp += fmt.Sprintf("<ctype=%d>", v.Ctype)
	return fp
}

// Fmt "%Z": escaped string literals
func Zconv(sp *Strlit, flag int) string {
	if sp == nil {
		var fp string
		fp += "<nil>"
		return fp
	}

	// NOTE: Keep in sync with ../ld/go.c:/^Zconv.
	s := sp.S
	var n int
	var fp string
	for i := 0; i < len(s); i += n {
		var r rune
		r, n = utf8.DecodeRuneInString(s[i:])
		switch r {
		case utf8.RuneError:
			if n == 1 {
				fp += fmt.Sprintf("\\x%02x", s[i])
				break
			}
			fallthrough

			// fall through
		default:
			if r < ' ' {
				fp += fmt.Sprintf("\\x%02x", r)
				break
			}

			fp += string(r)

		case '\t':
			fp += "\\t"

		case '\n':
			fp += "\\n"

		case '"',
			'\\':
			fp += `\` + string(r)

		case 0xFEFF: // BOM, basically disallowed in source code
			fp += "\\uFEFF"
		}
	}

	return fp
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
func Econv(et int, flag int) string {
	if et >= 0 && et < len(etnames) && etnames[et] != "" {
		var fp string
		fp += etnames[et]
		return fp
	}
	var fp string
	fp += fmt.Sprintf("E-%d", et)
	return fp
}

// Fmt "%S": syms
func symfmt(s *Sym, flag int) string {
	if s.Pkg != nil && flag&obj.FmtShort == 0 /*untyped*/ {
		switch fmtmode {
		case FErr: // This is for the user
			if s.Pkg == localpkg {
				var fp string
				fp += s.Name
				return fp
			}

			// If the name was used by multiple packages, display the full path,
			if s.Pkg.Name != "" && Pkglookup(s.Pkg.Name, nil).Npkg > 1 {
				var fp string
				fp += fmt.Sprintf("\"%v\".%s", Zconv(s.Pkg.Path, 0), s.Name)
				return fp
			}
			var fp string
			fp += fmt.Sprintf("%s.%s", s.Pkg.Name, s.Name)
			return fp

		case FDbg:
			var fp string
			fp += fmt.Sprintf("%s.%s", s.Pkg.Name, s.Name)
			return fp

		case FTypeId:
			if flag&obj.FmtUnsigned != 0 /*untyped*/ {
				var fp string
				fp += fmt.Sprintf("%s.%s", s.Pkg.Name, s.Name)
				return fp // dcommontype, typehash
			}
			var fp string
			fp += fmt.Sprintf("%s.%s", s.Pkg.Prefix, s.Name)
			return fp // (methodsym), typesym, weaksym

		case FExp:
			if s.Name != "" && s.Name[0] == '.' {
				Fatal("exporting synthetic symbol %s", s.Name)
			}
			if s.Pkg != builtinpkg {
				var fp string
				fp += fmt.Sprintf("@\"%v\".%s", Zconv(s.Pkg.Path, 0), s.Name)
				return fp
			}
		}
	}

	if flag&obj.FmtByte != 0 /*untyped*/ { // FmtByte (hh) implies FmtShort (h)

		// skip leading "type." in method name
		p := s.Name
		if i := strings.LastIndex(s.Name, "."); i >= 0 {
			p = s.Name[i+1:]
		}

		// exportname needs to see the name without the prefix too.
		if (fmtmode == FExp && !exportname(p)) || fmtmode == FDbg {
			var fp string
			fp += fmt.Sprintf("@\"%v\".%s", Zconv(s.Pkg.Path, 0), p)
			return fp
		}

		var fp string
		fp += p
		return fp
	}

	var fp string
	fp += s.Name
	return fp
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
		var fp string
		fp += "<T>"
		return fp
	}

	if t == bytetype || t == runetype {
		// in %-T mode collapse rune and byte with their originals.
		if fmtmode != FTypeId {
			var fp string
			fp += fmt.Sprintf("%v", Sconv(t.Sym, obj.FmtShort))
			return fp
		}
		t = Types[t.Etype]
	}

	if t == errortype {
		var fp string
		fp += "error"
		return fp
	}

	// Unless the 'l' flag was specified, if the type has a name, just print that name.
	if flag&obj.FmtLong == 0 /*untyped*/ && t.Sym != nil && t.Etype != TFIELD && t != Types[t.Etype] {
		switch fmtmode {
		case FTypeId:
			if flag&obj.FmtShort != 0 /*untyped*/ {
				if t.Vargen != 0 {
					var fp string
					fp += fmt.Sprintf("%v·%d", Sconv(t.Sym, obj.FmtShort), t.Vargen)
					return fp
				}
				var fp string
				fp += fmt.Sprintf("%v", Sconv(t.Sym, obj.FmtShort))
				return fp
			}

			if flag&obj.FmtUnsigned != 0 /*untyped*/ {
				var fp string
				fp += fmt.Sprintf("%v", Sconv(t.Sym, obj.FmtUnsigned))
				return fp
			}
			fallthrough

			// fallthrough
		case FExp:
			if t.Sym.Pkg == localpkg && t.Vargen != 0 {
				var fp string
				fp += fmt.Sprintf("%v·%d", Sconv(t.Sym, 0), t.Vargen)
				return fp
			}
		}

		var fp string
		fp += fmt.Sprintf("%v", Sconv(t.Sym, 0))
		return fp
	}

	var fp string
	if int(t.Etype) < len(basicnames) && basicnames[t.Etype] != "" {
		if fmtmode == FErr && (t == idealbool || t == idealstring) {
			fp += "untyped "
		}
		fp += basicnames[t.Etype]
		return fp
	}

	if fmtmode == FDbg {
		fp += fmt.Sprintf("%v-", Econv(int(t.Etype), 0))
	}

	switch t.Etype {
	case TPTR32,
		TPTR64:
		if fmtmode == FTypeId && (flag&obj.FmtShort != 0 /*untyped*/) {
			fp += fmt.Sprintf("*%v", Tconv(t.Type, obj.FmtShort))
			return fp
		}
		fp += fmt.Sprintf("*%v", Tconv(t.Type, 0))
		return fp

	case TARRAY:
		if t.Bound >= 0 {
			fp += fmt.Sprintf("[%d]%v", t.Bound, Tconv(t.Type, 0))
			return fp
		}
		if t.Bound == -100 {
			fp += fmt.Sprintf("[...]%v", Tconv(t.Type, 0))
			return fp
		}
		fp += fmt.Sprintf("[]%v", Tconv(t.Type, 0))
		return fp

	case TCHAN:
		switch t.Chan {
		case Crecv:
			fp += fmt.Sprintf("<-chan %v", Tconv(t.Type, 0))
			return fp

		case Csend:
			fp += fmt.Sprintf("chan<- %v", Tconv(t.Type, 0))
			return fp
		}

		if t.Type != nil && t.Type.Etype == TCHAN && t.Type.Sym == nil && t.Type.Chan == Crecv {
			fp += fmt.Sprintf("chan (%v)", Tconv(t.Type, 0))
			return fp
		}
		fp += fmt.Sprintf("chan %v", Tconv(t.Type, 0))
		return fp

	case TMAP:
		fp += fmt.Sprintf("map[%v]%v", Tconv(t.Down, 0), Tconv(t.Type, 0))
		return fp

	case TINTER:
		fp += "interface {"
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			if exportname(t1.Sym.Name) {
				if t1.Down != nil {
					fp += fmt.Sprintf(" %v%v;", Sconv(t1.Sym, obj.FmtShort), Tconv(t1.Type, obj.FmtShort))
				} else {
					fp += fmt.Sprintf(" %v%v ", Sconv(t1.Sym, obj.FmtShort), Tconv(t1.Type, obj.FmtShort))
				}
			} else {
				// non-exported method names must be qualified
				if t1.Down != nil {
					fp += fmt.Sprintf(" %v%v;", Sconv(t1.Sym, obj.FmtUnsigned), Tconv(t1.Type, obj.FmtShort))
				} else {
					fp += fmt.Sprintf(" %v%v ", Sconv(t1.Sym, obj.FmtUnsigned), Tconv(t1.Type, obj.FmtShort))
				}
			}
		}

		fp += "}"
		return fp

	case TFUNC:
		if flag&obj.FmtShort != 0 /*untyped*/ {
			fp += fmt.Sprintf("%v", Tconv(getinargx(t), 0))
		} else {
			if t.Thistuple != 0 {
				fp += fmt.Sprintf("method%v func%v", Tconv(getthisx(t), 0), Tconv(getinargx(t), 0))
			} else {
				fp += fmt.Sprintf("func%v", Tconv(getinargx(t), 0))
			}
		}

		switch t.Outtuple {
		case 0:
			break

		case 1:
			if fmtmode != FExp {
				fp += fmt.Sprintf(" %v", Tconv(getoutargx(t).Type.Type, 0)) // struct->field->field's type
				break
			}
			fallthrough

		default:
			fp += fmt.Sprintf(" %v", Tconv(getoutargx(t), 0))
		}

		return fp

		// Format the bucket struct for map[x]y as map.bucket[x]y.
	// This avoids a recursive print that generates very long names.
	case TSTRUCT:
		if t.Map != nil {
			if t.Map.Bucket == t {
				fp += fmt.Sprintf("map.bucket[%v]%v", Tconv(t.Map.Down, 0), Tconv(t.Map.Type, 0))
				return fp
			}

			if t.Map.Hmap == t {
				fp += fmt.Sprintf("map.hdr[%v]%v", Tconv(t.Map.Down, 0), Tconv(t.Map.Type, 0))
				return fp
			}

			if t.Map.Hiter == t {
				fp += fmt.Sprintf("map.iter[%v]%v", Tconv(t.Map.Down, 0), Tconv(t.Map.Type, 0))
				return fp
			}

			Yyerror("unknown internal map type")
		}

		if t.Funarg != 0 {
			fp += "("
			if fmtmode == FTypeId || fmtmode == FErr { // no argument names on function signature, and no "noescape"/"nosplit" tags
				for t1 := t.Type; t1 != nil; t1 = t1.Down {
					if t1.Down != nil {
						fp += fmt.Sprintf("%v, ", Tconv(t1, obj.FmtShort))
					} else {
						fp += fmt.Sprintf("%v", Tconv(t1, obj.FmtShort))
					}
				}
			} else {
				for t1 := t.Type; t1 != nil; t1 = t1.Down {
					if t1.Down != nil {
						fp += fmt.Sprintf("%v, ", Tconv(t1, 0))
					} else {
						fp += fmt.Sprintf("%v", Tconv(t1, 0))
					}
				}
			}

			fp += ")"
		} else {
			fp += "struct {"
			for t1 := t.Type; t1 != nil; t1 = t1.Down {
				if t1.Down != nil {
					fp += fmt.Sprintf(" %v;", Tconv(t1, obj.FmtLong))
				} else {
					fp += fmt.Sprintf(" %v ", Tconv(t1, obj.FmtLong))
				}
			}
			fp += "}"
		}

		return fp

	case TFIELD:
		if flag&obj.FmtShort == 0 /*untyped*/ {
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
				if t.Funarg != 0 {
					fp += fmt.Sprintf("%v ", Nconv(t.Nname, 0))
				} else if flag&obj.FmtLong != 0 /*untyped*/ {
					fp += fmt.Sprintf("%v ", Sconv(s, obj.FmtShort|obj.FmtByte)) // qualify non-exported names (used on structs, not on funarg)
				} else {
					fp += fmt.Sprintf("%v ", Sconv(s, 0))
				}
			} else if fmtmode == FExp {
				// TODO(rsc) this breaks on the eliding of unused arguments in the backend
				// when this is fixed, the special case in dcl.c checkarglist can go.
				//if(t->funarg)
				//	fmtstrcpy(fp, "_ ");
				//else
				if t.Embedded != 0 && s.Pkg != nil && len(s.Pkg.Path.S) > 0 {
					fp += fmt.Sprintf("@\"%v\".? ", Zconv(s.Pkg.Path, 0))
				} else {
					fp += "? "
				}
			}
		}

		if t.Isddd != 0 {
			fp += fmt.Sprintf("...%v", Tconv(t.Type.Type, 0))
		} else {
			fp += fmt.Sprintf("%v", Tconv(t.Type, 0))
		}

		if flag&obj.FmtShort == 0 /*untyped*/ && t.Note != nil {
			fp += fmt.Sprintf(" \"%v\"", Zconv(t.Note, 0))
		}
		return fp

	case TFORW:
		if t.Sym != nil {
			fp += fmt.Sprintf("undefined %v", Sconv(t.Sym, 0))
			return fp
		}
		fp += "undefined"
		return fp

	case TUNSAFEPTR:
		if fmtmode == FExp {
			fp += fmt.Sprintf("@\"unsafe\".Pointer")
			return fp
		}
		fp += fmt.Sprintf("unsafe.Pointer")
		return fp
	}

	if fmtmode == FExp {
		Fatal("missing %v case during export", Econv(int(t.Etype), 0))
	}

	// Don't know how to handle - fall back to detailed prints.
	fp += fmt.Sprintf("%v <%v> %v", Econv(int(t.Etype), 0), Sconv(t.Sym, 0), Tconv(t.Type, 0))
	return fp
}

// Statements which may be rendered with a simplestmt as init.
func stmtwithinit(op int) bool {
	switch op {
	case OIF,
		OFOR,
		OSWITCH:
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
	simpleinit := n.Ninit != nil && n.Ninit.Next == nil && n.Ninit.N.Ninit == nil && stmtwithinit(int(n.Op))

	// otherwise, print the inits as separate statements
	complexinit := n.Ninit != nil && !simpleinit && (fmtmode != FErr)

	// but if it was for if/for/switch, put in an extra surrounding block to limit the scope
	extrablock := complexinit && stmtwithinit(int(n.Op))

	if extrablock {
		f += "{"
	}

	if complexinit {
		f += fmt.Sprintf(" %v; ", Hconv(n.Ninit, 0))
	}

	switch n.Op {
	case ODCL:
		if fmtmode == FExp {
			switch n.Left.Class &^ PHEAP {
			case PPARAM,
				PPARAMOUT,
				PAUTO:
				f += fmt.Sprintf("var %v %v", Nconv(n.Left, 0), Tconv(n.Left.Type, 0))
				goto ret
			}
		}

		f += fmt.Sprintf("var %v %v", Sconv(n.Left.Sym, 0), Tconv(n.Left.Type, 0))

	case ODCLFIELD:
		if n.Left != nil {
			f += fmt.Sprintf("%v %v", Nconv(n.Left, 0), Nconv(n.Right, 0))
		} else {
			f += fmt.Sprintf("%v", Nconv(n.Right, 0))
		}

		// Don't export "v = <N>" initializing statements, hope they're always
	// preceded by the DCL which will be re-parsed and typecheck to reproduce
	// the "v = <N>" again.
	case OAS:
		if fmtmode == FExp && n.Right == nil {
			break
		}

		if n.Colas != 0 && !complexinit {
			f += fmt.Sprintf("%v := %v", Nconv(n.Left, 0), Nconv(n.Right, 0))
		} else {
			f += fmt.Sprintf("%v = %v", Nconv(n.Left, 0), Nconv(n.Right, 0))
		}

	case OASOP:
		if n.Implicit != 0 {
			if n.Etype == OADD {
				f += fmt.Sprintf("%v++", Nconv(n.Left, 0))
			} else {
				f += fmt.Sprintf("%v--", Nconv(n.Left, 0))
			}
			break
		}

		f += fmt.Sprintf("%v %v= %v", Nconv(n.Left, 0), Oconv(int(n.Etype), obj.FmtSharp), Nconv(n.Right, 0))

	case OAS2:
		if n.Colas != 0 && !complexinit {
			f += fmt.Sprintf("%v := %v", Hconv(n.List, obj.FmtComma), Hconv(n.Rlist, obj.FmtComma))
			break
		}
		fallthrough

		// fallthrough
	case OAS2DOTTYPE,
		OAS2FUNC,
		OAS2MAPR,
		OAS2RECV:
		f += fmt.Sprintf("%v = %v", Hconv(n.List, obj.FmtComma), Hconv(n.Rlist, obj.FmtComma))

	case ORETURN:
		f += fmt.Sprintf("return %v", Hconv(n.List, obj.FmtComma))

	case ORETJMP:
		f += fmt.Sprintf("retjmp %v", Sconv(n.Sym, 0))

	case OPROC:
		f += fmt.Sprintf("go %v", Nconv(n.Left, 0))

	case ODEFER:
		f += fmt.Sprintf("defer %v", Nconv(n.Left, 0))

	case OIF:
		if simpleinit {
			f += fmt.Sprintf("if %v; %v { %v }", Nconv(n.Ninit.N, 0), Nconv(n.Ntest, 0), Hconv(n.Nbody, 0))
		} else {
			f += fmt.Sprintf("if %v { %v }", Nconv(n.Ntest, 0), Hconv(n.Nbody, 0))
		}
		if n.Nelse != nil {
			f += fmt.Sprintf(" else { %v }", Hconv(n.Nelse, 0))
		}

	case OFOR:
		if fmtmode == FErr { // TODO maybe only if FmtShort, same below
			f += "for loop"
			break
		}

		f += "for"
		if simpleinit {
			f += fmt.Sprintf(" %v;", Nconv(n.Ninit.N, 0))
		} else if n.Nincr != nil {
			f += " ;"
		}

		if n.Ntest != nil {
			f += fmt.Sprintf(" %v", Nconv(n.Ntest, 0))
		}

		if n.Nincr != nil {
			f += fmt.Sprintf("; %v", Nconv(n.Nincr, 0))
		} else if simpleinit {
			f += ";"
		}

		f += fmt.Sprintf(" { %v }", Hconv(n.Nbody, 0))

	case ORANGE:
		if fmtmode == FErr {
			f += "for loop"
			break
		}

		if n.List == nil {
			f += fmt.Sprintf("for range %v { %v }", Nconv(n.Right, 0), Hconv(n.Nbody, 0))
			break
		}

		f += fmt.Sprintf("for %v = range %v { %v }", Hconv(n.List, obj.FmtComma), Nconv(n.Right, 0), Hconv(n.Nbody, 0))

	case OSELECT,
		OSWITCH:
		if fmtmode == FErr {
			f += fmt.Sprintf("%v statement", Oconv(int(n.Op), 0))
			break
		}

		f += fmt.Sprintf("%v", Oconv(int(n.Op), obj.FmtSharp))
		if simpleinit {
			f += fmt.Sprintf(" %v;", Nconv(n.Ninit.N, 0))
		}
		if n.Ntest != nil {
			f += fmt.Sprintf("%v", Nconv(n.Ntest, 0))
		}

		f += fmt.Sprintf(" { %v }", Hconv(n.List, 0))

	case OCASE,
		OXCASE:
		if n.List != nil {
			f += fmt.Sprintf("case %v: %v", Hconv(n.List, obj.FmtComma), Hconv(n.Nbody, 0))
		} else {
			f += fmt.Sprintf("default: %v", Hconv(n.Nbody, 0))
		}

	case OBREAK,
		OCONTINUE,
		OGOTO,
		OFALL,
		OXFALL:
		if n.Left != nil {
			f += fmt.Sprintf("%v %v", Oconv(int(n.Op), obj.FmtSharp), Nconv(n.Left, 0))
		} else {
			f += fmt.Sprintf("%v", Oconv(int(n.Op), obj.FmtSharp))
		}

	case OEMPTY:
		break

	case OLABEL:
		f += fmt.Sprintf("%v: ", Nconv(n.Left, 0))
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
	for n != nil && n.Implicit != 0 && (n.Op == OIND || n.Op == OADDR) {
		n = n.Left
	}

	if n == nil {
		var f string
		f += "<N>"
		return f
	}

	nprec := opprec[n.Op]
	if n.Op == OTYPE && n.Sym != nil {
		nprec = 8
	}

	if prec > nprec {
		var f string
		f += fmt.Sprintf("(%v)", Nconv(n, 0))
		return f
	}

	switch n.Op {
	case OPAREN:
		var f string
		f += fmt.Sprintf("(%v)", Nconv(n.Left, 0))
		return f

	case ODDDARG:
		var f string
		f += fmt.Sprintf("... argument")
		return f

	case OREGISTER:
		var f string
		f += fmt.Sprintf("%v", Ctxt.Rconv(int(n.Val.U.Reg)))
		return f

	case OLITERAL: // this is a bit of a mess
		if fmtmode == FErr && n.Sym != nil {
			var f string
			f += fmt.Sprintf("%v", Sconv(n.Sym, 0))
			return f
		}
		if n.Val.Ctype == CTNIL && n.Orig != nil && n.Orig != n {
			var f string
			f += exprfmt(n.Orig, prec)
			return f
		}
		if n.Type != nil && n.Type != Types[n.Type.Etype] && n.Type != idealbool && n.Type != idealstring {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if Isptr[n.Type.Etype] != 0 || (n.Type.Etype == TCHAN && n.Type.Chan == Crecv) {
				var f string
				f += fmt.Sprintf("(%v)(%v)", Tconv(n.Type, 0), Vconv(&n.Val, 0))
				return f
			} else {
				var f string
				f += fmt.Sprintf("%v(%v)", Tconv(n.Type, 0), Vconv(&n.Val, 0))
				return f
			}
		}

		var f string
		f += fmt.Sprintf("%v", Vconv(&n.Val, 0))
		return f

		// Special case: name used as local variable in export.
	// _ becomes ~b%d internally; print as _ for export
	case ONAME:
		if fmtmode == FExp && n.Sym != nil && n.Sym.Name[0] == '~' && n.Sym.Name[1] == 'b' {
			var f string
			f += fmt.Sprintf("_")
			return f
		}
		if fmtmode == FExp && n.Sym != nil && !isblank(n) && n.Vargen > 0 {
			var f string
			f += fmt.Sprintf("%v·%d", Sconv(n.Sym, 0), n.Vargen)
			return f
		}

		// Special case: explicit name of func (*T) method(...) is turned into pkg.(*T).method,
		// but for export, this should be rendered as (*pkg.T).meth.
		// These nodes have the special property that they are names with a left OTYPE and a right ONAME.
		if fmtmode == FExp && n.Left != nil && n.Left.Op == OTYPE && n.Right != nil && n.Right.Op == ONAME {
			if Isptr[n.Left.Type.Etype] != 0 {
				var f string
				f += fmt.Sprintf("(%v).%v", Tconv(n.Left.Type, 0), Sconv(n.Right.Sym, obj.FmtShort|obj.FmtByte))
				return f
			} else {
				var f string
				f += fmt.Sprintf("%v.%v", Tconv(n.Left.Type, 0), Sconv(n.Right.Sym, obj.FmtShort|obj.FmtByte))
				return f
			}
		}
		fallthrough

		//fallthrough
	case OPACK,
		ONONAME:
		var f string
		f += fmt.Sprintf("%v", Sconv(n.Sym, 0))
		return f

	case OTYPE:
		if n.Type == nil && n.Sym != nil {
			var f string
			f += fmt.Sprintf("%v", Sconv(n.Sym, 0))
			return f
		}
		var f string
		f += fmt.Sprintf("%v", Tconv(n.Type, 0))
		return f

	case OTARRAY:
		if n.Left != nil {
			var f string
			f += fmt.Sprintf("[]%v", Nconv(n.Left, 0))
			return f
		}
		var f string
		f += fmt.Sprintf("[]%v", Nconv(n.Right, 0))
		return f // happens before typecheck

	case OTMAP:
		var f string
		f += fmt.Sprintf("map[%v]%v", Nconv(n.Left, 0), Nconv(n.Right, 0))
		return f

	case OTCHAN:
		switch n.Etype {
		case Crecv:
			var f string
			f += fmt.Sprintf("<-chan %v", Nconv(n.Left, 0))
			return f

		case Csend:
			var f string
			f += fmt.Sprintf("chan<- %v", Nconv(n.Left, 0))
			return f

		default:
			if n.Left != nil && n.Left.Op == OTCHAN && n.Left.Sym == nil && n.Left.Etype == Crecv {
				var f string
				f += fmt.Sprintf("chan (%v)", Nconv(n.Left, 0))
				return f
			} else {
				var f string
				f += fmt.Sprintf("chan %v", Nconv(n.Left, 0))
				return f
			}
		}
		fallthrough

	case OTSTRUCT:
		var f string
		f += fmt.Sprintf("<struct>")
		return f

	case OTINTER:
		var f string
		f += fmt.Sprintf("<inter>")
		return f

	case OTFUNC:
		var f string
		f += fmt.Sprintf("<func>")
		return f

	case OCLOSURE:
		if fmtmode == FErr {
			var f string
			f += "func literal"
			return f
		}
		if n.Nbody != nil {
			var f string
			f += fmt.Sprintf("%v { %v }", Tconv(n.Type, 0), Hconv(n.Nbody, 0))
			return f
		}
		var f string
		f += fmt.Sprintf("%v { %v }", Tconv(n.Type, 0), Hconv(n.Closure.Nbody, 0))
		return f

	case OCOMPLIT:
		ptrlit := n.Right != nil && n.Right.Implicit != 0 && n.Right.Type != nil && Isptr[n.Right.Type.Etype] != 0
		if fmtmode == FErr {
			if n.Right != nil && n.Right.Type != nil && n.Implicit == 0 {
				if ptrlit {
					var f string
					f += fmt.Sprintf("&%v literal", Tconv(n.Right.Type.Type, 0))
					return f
				} else {
					var f string
					f += fmt.Sprintf("%v literal", Tconv(n.Right.Type, 0))
					return f
				}
			}

			var f string
			f += "composite literal"
			return f
		}

		if fmtmode == FExp && ptrlit {
			// typecheck has overwritten OIND by OTYPE with pointer type.
			var f string
			f += fmt.Sprintf("(&%v{ %v })", Tconv(n.Right.Type.Type, 0), Hconv(n.List, obj.FmtComma))
			return f
		}

		var f string
		f += fmt.Sprintf("(%v{ %v })", Nconv(n.Right, 0), Hconv(n.List, obj.FmtComma))
		return f

	case OPTRLIT:
		if fmtmode == FExp && n.Left.Implicit != 0 {
			var f string
			f += fmt.Sprintf("%v", Nconv(n.Left, 0))
			return f
		}
		var f string
		f += fmt.Sprintf("&%v", Nconv(n.Left, 0))
		return f

	case OSTRUCTLIT:
		if fmtmode == FExp { // requires special handling of field names
			var f string
			if n.Implicit != 0 {
				f += "{"
			} else {
				f += fmt.Sprintf("(%v{", Tconv(n.Type, 0))
			}
			for l := n.List; l != nil; l = l.Next {
				f += fmt.Sprintf(" %v:%v", Sconv(l.N.Left.Sym, obj.FmtShort|obj.FmtByte), Nconv(l.N.Right, 0))

				if l.Next != nil {
					f += ","
				} else {
					f += " "
				}
			}

			if n.Implicit == 0 {
				f += "})"
				return f
			}
			f += "}"
			return f
		}
		fallthrough

		// fallthrough

	case OARRAYLIT,
		OMAPLIT:
		if fmtmode == FErr {
			var f string
			f += fmt.Sprintf("%v literal", Tconv(n.Type, 0))
			return f
		}
		if fmtmode == FExp && n.Implicit != 0 {
			var f string
			f += fmt.Sprintf("{ %v }", Hconv(n.List, obj.FmtComma))
			return f
		}
		var f string
		f += fmt.Sprintf("(%v{ %v })", Tconv(n.Type, 0), Hconv(n.List, obj.FmtComma))
		return f

	case OKEY:
		if n.Left != nil && n.Right != nil {
			if fmtmode == FExp && n.Left.Type != nil && n.Left.Type.Etype == TFIELD {
				// requires special handling of field names
				var f string
				f += fmt.Sprintf("%v:%v", Sconv(n.Left.Sym, obj.FmtShort|obj.FmtByte), Nconv(n.Right, 0))
				return f
			} else {
				var f string
				f += fmt.Sprintf("%v:%v", Nconv(n.Left, 0), Nconv(n.Right, 0))
				return f
			}
		}

		if n.Left == nil && n.Right != nil {
			var f string
			f += fmt.Sprintf(":%v", Nconv(n.Right, 0))
			return f
		}
		if n.Left != nil && n.Right == nil {
			var f string
			f += fmt.Sprintf("%v:", Nconv(n.Left, 0))
			return f
		}
		var f string
		f += ":"
		return f

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

	case ODOTTYPE,
		ODOTTYPE2:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Right != nil {
			f += fmt.Sprintf(".(%v)", Nconv(n.Right, 0))
			return f
		}
		f += fmt.Sprintf(".(%v)", Tconv(n.Type, 0))
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
		f += fmt.Sprintf("[%v]", Nconv(n.Right, 0))
		return f

	case OCOPY,
		OCOMPLEX:
		var f string
		f += fmt.Sprintf("%v(%v, %v)", Oconv(int(n.Op), obj.FmtSharp), Nconv(n.Left, 0), Nconv(n.Right, 0))
		return f

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		ORUNESTR:
		if n.Type == nil || n.Type.Sym == nil {
			var f string
			f += fmt.Sprintf("(%v)(%v)", Tconv(n.Type, 0), Nconv(n.Left, 0))
			return f
		}
		if n.Left != nil {
			var f string
			f += fmt.Sprintf("%v(%v)", Tconv(n.Type, 0), Nconv(n.Left, 0))
			return f
		}
		var f string
		f += fmt.Sprintf("%v(%v)", Tconv(n.Type, 0), Hconv(n.List, obj.FmtComma))
		return f

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
			var f string
			f += fmt.Sprintf("%v(%v)", Oconv(int(n.Op), obj.FmtSharp), Nconv(n.Left, 0))
			return f
		}
		if n.Isddd != 0 {
			var f string
			f += fmt.Sprintf("%v(%v...)", Oconv(int(n.Op), obj.FmtSharp), Hconv(n.List, obj.FmtComma))
			return f
		}
		var f string
		f += fmt.Sprintf("%v(%v)", Oconv(int(n.Op), obj.FmtSharp), Hconv(n.List, obj.FmtComma))
		return f

	case OCALL,
		OCALLFUNC,
		OCALLINTER,
		OCALLMETH:
		var f string
		f += exprfmt(n.Left, nprec)
		if n.Isddd != 0 {
			f += fmt.Sprintf("(%v...)", Hconv(n.List, obj.FmtComma))
			return f
		}
		f += fmt.Sprintf("(%v)", Hconv(n.List, obj.FmtComma))
		return f

	case OMAKEMAP,
		OMAKECHAN,
		OMAKESLICE:
		if n.List != nil { // pre-typecheck
			var f string
			f += fmt.Sprintf("make(%v, %v)", Tconv(n.Type, 0), Hconv(n.List, obj.FmtComma))
			return f
		}
		if n.Right != nil {
			var f string
			f += fmt.Sprintf("make(%v, %v, %v)", Tconv(n.Type, 0), Nconv(n.Left, 0), Nconv(n.Right, 0))
			return f
		}
		if n.Left != nil {
			var f string
			f += fmt.Sprintf("make(%v, %v)", Tconv(n.Type, 0), Nconv(n.Left, 0))
			return f
		}
		var f string
		f += fmt.Sprintf("make(%v)", Tconv(n.Type, 0))
		return f

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
			f += fmt.Sprintf("%v", Oconv(int(n.Op), obj.FmtSharp))
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
				f += fmt.Sprintf(" + ")
			}
			f += exprfmt(l.N, nprec)
		}

		return f

	case OCMPSTR,
		OCMPIFACE:
		var f string
		f += exprfmt(n.Left, nprec)
		f += fmt.Sprintf(" %v ", Oconv(int(n.Etype), obj.FmtSharp))
		f += exprfmt(n.Right, nprec+1)
		return f
	}

	var f string
	f += fmt.Sprintf("<node %v>", Oconv(int(n.Op), 0))
	return f
}

func nodefmt(n *Node, flag int) string {
	t := n.Type

	// we almost always want the original, except in export mode for literals
	// this saves the importer some work, and avoids us having to redo some
	// special casing for package unsafe
	if (fmtmode != FExp || n.Op != OLITERAL) && n.Orig != nil {
		n = n.Orig
	}

	if flag&obj.FmtLong != 0 /*untyped*/ && t != nil {
		if t.Etype == TNIL {
			var f string
			f += fmt.Sprintf("nil")
			return f
		} else {
			var f string
			f += fmt.Sprintf("%v (type %v)", Nconv(n, 0), Tconv(t, 0))
			return f
		}
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if opprec[n.Op] < 0 {
		return stmtfmt(n)
	}

	var f string
	f += exprfmt(n, 0)
	return f
}

var dumpdepth int

func indent(s string) string {
	return s + "\n" + strings.Repeat(".   ", dumpdepth)
}

func nodedump(n *Node, flag int) string {
	if n == nil {
		var fp string
		return fp
	}

	recur := flag&obj.FmtShort == 0 /*untyped*/

	var fp string
	if recur {
		fp = indent(fp)
		if dumpdepth > 10 {
			fp += "..."
			return fp
		}

		if n.Ninit != nil {
			fp += fmt.Sprintf("%v-init%v", Oconv(int(n.Op), 0), Hconv(n.Ninit, 0))
			fp = indent(fp)
		}
	}

	//	fmtprint(fp, "[%p]", n);

	switch n.Op {
	default:
		fp += fmt.Sprintf("%v%v", Oconv(int(n.Op), 0), Jconv(n, 0))

	case OREGISTER,
		OINDREG:
		fp += fmt.Sprintf("%v-%v%v", Oconv(int(n.Op), 0), Ctxt.Rconv(int(n.Val.U.Reg)), Jconv(n, 0))

	case OLITERAL:
		fp += fmt.Sprintf("%v-%v%v", Oconv(int(n.Op), 0), Vconv(&n.Val, 0), Jconv(n, 0))

	case ONAME,
		ONONAME:
		if n.Sym != nil {
			fp += fmt.Sprintf("%v-%v%v", Oconv(int(n.Op), 0), Sconv(n.Sym, 0), Jconv(n, 0))
		} else {
			fp += fmt.Sprintf("%v%v", Oconv(int(n.Op), 0), Jconv(n, 0))
		}
		if recur && n.Type == nil && n.Ntype != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-ntype%v", Oconv(int(n.Op), 0), Nconv(n.Ntype, 0))
		}

	case OASOP:
		fp += fmt.Sprintf("%v-%v%v", Oconv(int(n.Op), 0), Oconv(int(n.Etype), 0), Jconv(n, 0))

	case OTYPE:
		fp += fmt.Sprintf("%v %v%v type=%v", Oconv(int(n.Op), 0), Sconv(n.Sym, 0), Jconv(n, 0), Tconv(n.Type, 0))
		if recur && n.Type == nil && n.Ntype != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-ntype%v", Oconv(int(n.Op), 0), Nconv(n.Ntype, 0))
		}
	}

	if n.Sym != nil && n.Op != ONAME {
		fp += fmt.Sprintf(" %v G%d", Sconv(n.Sym, 0), n.Vargen)
	}

	if n.Type != nil {
		fp += fmt.Sprintf(" %v", Tconv(n.Type, 0))
	}

	if recur {
		if n.Left != nil {
			fp += fmt.Sprintf("%v", Nconv(n.Left, 0))
		}
		if n.Right != nil {
			fp += fmt.Sprintf("%v", Nconv(n.Right, 0))
		}
		if n.List != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-list%v", Oconv(int(n.Op), 0), Hconv(n.List, 0))
		}

		if n.Rlist != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-rlist%v", Oconv(int(n.Op), 0), Hconv(n.Rlist, 0))
		}

		if n.Ntest != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-test%v", Oconv(int(n.Op), 0), Nconv(n.Ntest, 0))
		}

		if n.Nbody != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-body%v", Oconv(int(n.Op), 0), Hconv(n.Nbody, 0))
		}

		if n.Nelse != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-else%v", Oconv(int(n.Op), 0), Hconv(n.Nelse, 0))
		}

		if n.Nincr != nil {
			fp = indent(fp)
			fp += fmt.Sprintf("%v-incr%v", Oconv(int(n.Op), 0), Nconv(n.Nincr, 0))
		}
	}

	return fp
}

// Fmt "%S": syms
// Flags:  "%hS" suppresses qualifying with package
func Sconv(s *Sym, flag int) string {
	if flag&obj.FmtLong != 0 /*untyped*/ {
		panic("linksymfmt")
	}

	if s == nil {
		var fp string
		fp += "<S>"
		return fp
	}

	if s.Name == "_" {
		var fp string
		fp += "_"
		return fp
	}

	sf := flag
	sm := setfmode(&flag)
	var r int
	_ = r
	str := symfmt(s, flag)
	flag = sf
	fmtmode = sm
	return str
}

// Fmt "%T": types.
// Flags: 'l' print definition, not name
//	  'h' omit 'func' and receiver from function types, short type names
//	  'u' package name, not prefix (FTypeId mode, sticky)
func Tconv(t *Type, flag int) string {
	if t == nil {
		var fp string
		fp += "<T>"
		return fp
	}

	if t.Trecur > 4 {
		var fp string
		fp += "<...>"
		return fp
	}

	t.Trecur++
	sf := flag
	sm := setfmode(&flag)

	if fmtmode == FTypeId && (sf&obj.FmtUnsigned != 0) {
		fmtpkgpfx++
	}
	if fmtpkgpfx != 0 {
		flag |= obj.FmtUnsigned
	}

	var r int
	_ = r
	str := typefmt(t, flag)

	if fmtmode == FTypeId && (sf&obj.FmtUnsigned != 0) {
		fmtpkgpfx--
	}

	flag = sf
	fmtmode = sm
	t.Trecur--
	return str
}

// Fmt '%N': Nodes.
// Flags: 'l' suffix with "(type %T)" where possible
//	  '+h' in debug mode, don't recurse, no multiline output
func Nconv(n *Node, flag int) string {
	if n == nil {
		var fp string
		fp += "<N>"
		return fp
	}
	sf := flag
	sm := setfmode(&flag)

	var r int
	_ = r
	var str string
	switch fmtmode {
	case FErr,
		FExp:
		str = nodefmt(n, flag)

	case FDbg:
		dumpdepth++
		str = nodedump(n, flag)
		dumpdepth--

	default:
		Fatal("unhandled %%N mode")
	}

	flag = sf
	fmtmode = sm
	return str
}

// Fmt '%H': NodeList.
// Flags: all those of %N plus ',': separate with comma's instead of semicolons.
func Hconv(l *NodeList, flag int) string {
	if l == nil && fmtmode == FDbg {
		var fp string
		fp += "<nil>"
		return fp
	}

	sf := flag
	sm := setfmode(&flag)
	var r int
	_ = r
	sep := "; "
	if fmtmode == FDbg {
		sep = "\n"
	} else if flag&obj.FmtComma != 0 /*untyped*/ {
		sep = ", "
	}

	var fp string
	for ; l != nil; l = l.Next {
		fp += fmt.Sprintf("%v", Nconv(l.N, 0))
		if l.Next != nil {
			fp += sep
		}
	}

	flag = sf
	fmtmode = sm
	return fp
}

func dumplist(s string, l *NodeList) {
	fmt.Printf("%s%v\n", s, Hconv(l, obj.FmtSign))
}

func Dump(s string, n *Node) {
	fmt.Printf("%s [%p]%v\n", s, n, Nconv(n, obj.FmtSign))
}
