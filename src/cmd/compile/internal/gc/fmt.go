// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// A FmtFlag value is a set of flags (or 0).
// They control how the Xconv functions format their values.
// See the respective function's documentation for details.
type FmtFlag int

// TODO(gri) The ' ' flag is not used anymore in %-formats.
//           Eliminate eventually.

const ( //                                 fmt.Format flag/prec or verb
	FmtLeft     FmtFlag = 1 << iota // '-'
	FmtSharp                        // '#'
	FmtSign                         // '+'
	FmtUnsigned                     // ' '               (historic: u flag)
	FmtShort                        // verb == 'S'       (historic: h flag)
	FmtLong                         // verb == 'L'       (historic: l flag)
	FmtComma                        // '.' (== hasPrec)  (historic: , flag)
	FmtByte                         // '0'               (historic: hh flag)
)

// fmtFlag computes the (internal) FmtFlag
// value given the fmt.State and format verb.
func fmtFlag(s fmt.State, verb rune) FmtFlag {
	var flag FmtFlag
	if s.Flag('-') {
		flag |= FmtLeft
	}
	if s.Flag('#') {
		flag |= FmtSharp
	}
	if s.Flag('+') {
		flag |= FmtSign
	}
	if s.Flag(' ') {
		flag |= FmtUnsigned
	}
	if _, ok := s.Precision(); ok {
		flag |= FmtComma
	}
	if s.Flag('0') {
		flag |= FmtByte
	}
	switch verb {
	case 'S':
		flag |= FmtShort
	case 'L':
		flag |= FmtLong
	}
	return flag
}

// Format conversions:
// TODO(gri) verify these; eliminate those not used anymore
//
//	%v Op		Node opcodes
//		Flags:  #: print Go syntax (automatic unless fmtmode == FDbg)
//
//	%j *Node	Node details
//		Flags:  0: suppresses things not relevant until walk
//
//	%v *Val		Constant values
//
//	%v *Sym		Symbols
//	%S              unqualified identifier in any mode
//		Flags:  +,- #: mode (see below)
//			0: in export mode: unqualified identifier if exported, qualified if not
//
//	%v *Type	Types
//	%S              omit "func" and receiver in function types
//	%L              definition instead of name.
//		Flags:  +,- #: mode (see below)
//			' ' (only in -/Sym mode) print type identifiers wit package name instead of prefix.
//
//	%v *Node	Nodes
//	%S              (only in +/debug mode) suppress recursion
//	%L              (only in Error mode) print "foo (type Bar)"
//		Flags:  +,- #: mode (see below)
//
//	%v Nodes	Node lists
//		Flags:  those of *Node
//			.: separate items with ',' instead of ';'

// *Sym, *Type, and *Node types use the flags below to set the format mode
const (
	FErr = iota
	FDbg
	FTypeId
)

var fmtmode int = FErr

var fmtpkgpfx int // "% v" stickyness for *Type objects

// The mode flags '+', '-', and '#' are sticky; they persist through
// recursions of *Node, *Type, and *Sym values. The ' ' flag is
// sticky only on *Type recursions and only used in %-/*Sym mode.
//
// Example: given a *Sym: %+v %#v %-v print an identifier properly qualified for debug/export/internal mode

// Useful format combinations:
// TODO(gri): verify these
//
// *Node, Nodes:
//   %+v    multiline recursive debug dump of *Node/Nodes
//   %+S    non-recursive debug dump
//
// *Node:
//   %#v    Go format
//   %L     "foo (type Bar)" for error messages
//
// *Type:
//   %#v    Go format
//   %#L    type definition instead of name
//   %#S    omit"func" and receiver in function signature
//
//   %-v    type identifiers
//   %-S    type identifiers without "func" and arg names in type signatures (methodsym)
//   %- v   type identifiers with package name instead of prefix (typesym, dcommontype, typehash)

func setfmode(flags *FmtFlag) (fm int) {
	fm = fmtmode
	if *flags&FmtSign != 0 {
		fmtmode = FDbg
	} else if *flags&FmtSharp != 0 {
		// ignore (textual export format no longer supported)
	} else if *flags&FmtLeft != 0 {
		fmtmode = FTypeId
	}

	*flags &^= (FmtSharp | FmtLeft | FmtSign)
	return
}

var goopnames = []string{
	OADDR:     "&",
	OADD:      "+",
	OADDSTR:   "+",
	OALIGNOF:  "unsafe.Alignof",
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
	OOFFSETOF: "unsafe.Offsetof",
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
	OSIZEOF:   "unsafe.Sizeof",
	OSUB:      "-",
	OSWITCH:   "switch",
	OXOR:      "^",
	OXFALL:    "fallthrough",
}

func (o Op) String() string {
	return fmt.Sprint(o)
}

func (o Op) GoString() string {
	return fmt.Sprintf("%#v", o)
}

func (o Op) Format(s fmt.State, verb rune) {
	switch verb {
	case 'v':
		o.oconv(s, fmtFlag(s, verb))

	default:
		fmt.Fprintf(s, "%%!%c(Op=%d)", verb, int(o))
	}
}

func (o Op) oconv(s fmt.State, flag FmtFlag) {
	if (flag&FmtSharp != 0) || fmtmode != FDbg {
		if o >= 0 && int(o) < len(goopnames) && goopnames[o] != "" {
			fmt.Fprint(s, goopnames[o])
			return
		}
	}

	if o >= 0 && int(o) < len(opnames) && opnames[o] != "" {
		fmt.Fprint(s, opnames[o])
		return
	}

	fmt.Fprintf(s, "O-%d", int(o))
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

func (n *Node) Format(s fmt.State, verb rune) {
	switch verb {
	case 'v', 'S', 'L':
		n.Nconv(s, fmtFlag(s, verb))

	case 'j':
		n.jconv(s, fmtFlag(s, verb))

	default:
		fmt.Fprintf(s, "%%!%c(*Node=%p)", verb, n)
	}
}

// *Node details
func (n *Node) jconv(s fmt.State, flag FmtFlag) {
	c := flag & FmtShort

	if c == 0 && n.Ullman != 0 {
		fmt.Fprintf(s, " u(%d)", n.Ullman)
	}

	if c == 0 && n.Addable {
		fmt.Fprintf(s, " a(%v)", n.Addable)
	}

	if c == 0 && n.Name != nil && n.Name.Vargen != 0 {
		fmt.Fprintf(s, " g(%d)", n.Name.Vargen)
	}

	if n.Lineno != 0 {
		fmt.Fprintf(s, " l(%d)", n.Lineno)
	}

	if c == 0 && n.Xoffset != BADWIDTH {
		fmt.Fprintf(s, " x(%d)", n.Xoffset)
	}

	if n.Class != 0 {
		if int(n.Class) < len(classnames) {
			fmt.Fprintf(s, " class(%s)", classnames[n.Class])
		} else {
			fmt.Fprintf(s, " class(%d?)", n.Class)
		}
	}

	if n.Colas {
		fmt.Fprintf(s, " colas(%v)", n.Colas)
	}

	if n.Name != nil && n.Name.Funcdepth != 0 {
		fmt.Fprintf(s, " f(%d)", n.Name.Funcdepth)
	}
	if n.Func != nil && n.Func.Depth != 0 {
		fmt.Fprintf(s, " ff(%d)", n.Func.Depth)
	}

	switch n.Esc {
	case EscUnknown:
		break

	case EscHeap:
		fmt.Fprint(s, " esc(h)")

	case EscNone:
		fmt.Fprint(s, " esc(no)")

	case EscNever:
		if c == 0 {
			fmt.Fprint(s, " esc(N)")
		}

	default:
		fmt.Fprintf(s, " esc(%d)", n.Esc)
	}

	if e, ok := n.Opt().(*NodeEscState); ok && e.Loopdepth != 0 {
		fmt.Fprintf(s, " ld(%d)", e.Loopdepth)
	}

	if c == 0 && n.Typecheck != 0 {
		fmt.Fprintf(s, " tc(%d)", n.Typecheck)
	}

	if c == 0 && n.IsStatic {
		fmt.Fprint(s, " static")
	}

	if n.Isddd {
		fmt.Fprintf(s, " isddd(%v)", n.Isddd)
	}

	if n.Implicit {
		fmt.Fprintf(s, " implicit(%v)", n.Implicit)
	}

	if n.Embedded != 0 {
		fmt.Fprintf(s, " embedded(%d)", n.Embedded)
	}

	if n.Addrtaken {
		fmt.Fprint(s, " addrtaken")
	}

	if n.Assigned {
		fmt.Fprint(s, " assigned")
	}
	if n.Bounded {
		fmt.Fprint(s, " bounded")
	}
	if n.NonNil {
		fmt.Fprint(s, " nonnil")
	}

	if c == 0 && n.Used {
		fmt.Fprintf(s, " used(%v)", n.Used)
	}
}

func (v Val) Format(s fmt.State, verb rune) {
	switch verb {
	case 'v':
		v.vconv(s, fmtFlag(s, verb))

	default:
		fmt.Fprintf(s, "%%!%c(Val=%T)", verb, v)
	}
}

func (v Val) vconv(s fmt.State, flag FmtFlag) {
	switch u := v.U.(type) {
	case *Mpint:
		if !u.Rune {
			if flag&FmtSharp != 0 {
				fmt.Fprint(s, bconv(u, FmtSharp))
				return
			}
			fmt.Fprint(s, bconv(u, 0))
			return
		}

		switch x := u.Int64(); {
		case ' ' <= x && x < utf8.RuneSelf && x != '\\' && x != '\'':
			fmt.Fprintf(s, "'%c'", int(x))

		case 0 <= x && x < 1<<16:
			fmt.Fprintf(s, "'\\u%04x'", uint(int(x)))

		case 0 <= x && x <= utf8.MaxRune:
			fmt.Fprintf(s, "'\\U%08x'", uint64(x))

		default:
			fmt.Fprintf(s, "('\\x00' + %v)", u)
		}

	case *Mpflt:
		if flag&FmtSharp != 0 {
			fmt.Fprint(s, fconv(u, 0))
			return
		}
		fmt.Fprint(s, fconv(u, FmtSharp))
		return

	case *Mpcplx:
		switch {
		case flag&FmtSharp != 0:
			fmt.Fprintf(s, "(%v+%vi)", &u.Real, &u.Imag)

		case v.U.(*Mpcplx).Real.CmpFloat64(0) == 0:
			fmt.Fprintf(s, "%vi", fconv(&u.Imag, FmtSharp))

		case v.U.(*Mpcplx).Imag.CmpFloat64(0) == 0:
			fmt.Fprint(s, fconv(&u.Real, FmtSharp))

		case v.U.(*Mpcplx).Imag.CmpFloat64(0) < 0:
			fmt.Fprintf(s, "(%v%vi)", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp))

		default:
			fmt.Fprintf(s, "(%v+%vi)", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp))
		}

	case string:
		fmt.Fprint(s, strconv.Quote(u))

	case bool:
		t := "false"
		if u {
			t = "true"
		}
		fmt.Fprint(s, t)

	case *NilVal:
		fmt.Fprint(s, "nil")

	default:
		fmt.Fprintf(s, "<ctype=%d>", v.Ctype())
	}
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

func (s *Sym) symfmt(flag FmtFlag) string {
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
			return fmt.Sprintf("@%q.%s", s.Pkg.Path, name)
		}

		return name
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

func (t *Type) typefmt(flag FmtFlag) string {
	if t == nil {
		return "<T>"
	}

	if t == bytetype || t == runetype {
		// in %-T mode collapse rune and byte with their originals.
		if fmtmode != FTypeId {
			return t.Sym.sconv(FmtShort)
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
					return fmt.Sprintf("%v·%d", t.Sym.sconv(FmtShort), t.Vargen)
				}
				return t.Sym.sconv(FmtShort)
			}

			if flag&FmtUnsigned != 0 {
				return t.Sym.sconv(FmtUnsigned)
			}

			if t.Sym.Pkg == localpkg && t.Vargen != 0 {
				return fmt.Sprintf("%v·%d", t.Sym, t.Vargen)
			}
		}

		return t.Sym.String()
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
		str := t.Etype.String() + "-" + t.typefmt(flag)
		fmtmode = FDbg
		return str
	}

	switch t.Etype {
	case TPTR32, TPTR64:
		if fmtmode == FTypeId && (flag&FmtShort != 0) {
			return "*" + t.Elem().tconv(FmtShort)
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
		if t.IsEmptyInterface() {
			return "interface {}"
		}
		buf := make([]byte, 0, 64)
		buf = append(buf, "interface {"...)
		for i, f := range t.Fields().Slice() {
			if i != 0 {
				buf = append(buf, ';')
			}
			buf = append(buf, ' ')
			switch {
			case f.Sym == nil:
				// Check first that a symbol is defined for this type.
				// Wrong interface definitions may have types lacking a symbol.
				break
			case exportname(f.Sym.Name):
				buf = append(buf, f.Sym.sconv(FmtShort)...)
			default:
				buf = append(buf, f.Sym.sconv(FmtUnsigned)...)
			}
			buf = append(buf, f.Type.tconv(FmtShort)...)
		}
		if t.NumFields() != 0 {
			buf = append(buf, ' ')
		}
		buf = append(buf, '}')
		return string(buf)

	case TFUNC:
		buf := make([]byte, 0, 64)
		if flag&FmtShort != 0 {
			// no leading func
		} else {
			if t.Recv() != nil {
				buf = append(buf, "method"...)
				buf = append(buf, t.Recvs().String()...)
				buf = append(buf, ' ')
			}
			buf = append(buf, "func"...)
		}
		buf = append(buf, t.Params().String()...)

		switch t.Results().NumFields() {
		case 0:
			// nothing to do

		case 1:
			buf = append(buf, ' ')
			buf = append(buf, t.Results().Field(0).Type.String()...) // struct->field->field's type

		default:
			buf = append(buf, ' ')
			buf = append(buf, t.Results().String()...)
		}
		return string(buf)

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

			yyerror("unknown internal map type")
		}

		buf := make([]byte, 0, 64)
		if t.IsFuncArgStruct() {
			buf = append(buf, '(')
			var flag1 FmtFlag
			if fmtmode == FTypeId || fmtmode == FErr { // no argument names on function signature, and no "noescape"/"nosplit" tags
				flag1 = FmtShort
			}
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					buf = append(buf, ", "...)
				}
				buf = append(buf, fldconv(f, flag1)...)
			}
			buf = append(buf, ')')
		} else {
			buf = append(buf, "struct {"...)
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					buf = append(buf, ';')
				}
				buf = append(buf, ' ')
				buf = append(buf, fldconv(f, FmtLong)...)
			}
			if t.NumFields() != 0 {
				buf = append(buf, ' ')
			}
			buf = append(buf, '}')
		}
		return string(buf)

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

func (n *Node) stmtfmt(s fmt.State) {
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
		fmt.Fprint(s, "{")
	}

	if complexinit {
		fmt.Fprintf(s, " %v; ", n.Ninit)
	}

	switch n.Op {
	case ODCL:
		fmt.Fprintf(s, "var %v %v", n.Left.Sym, n.Left.Type)

	case ODCLFIELD:
		if n.Left != nil {
			fmt.Fprintf(s, "%v %v", n.Left, n.Right)
		} else {
			fmt.Fprintf(s, "%v", n.Right)
		}

	// Don't export "v = <N>" initializing statements, hope they're always
	// preceded by the DCL which will be re-parsed and typechecked to reproduce
	// the "v = <N>" again.
	case OAS, OASWB:
		if n.Colas && !complexinit {
			fmt.Fprintf(s, "%v := %v", n.Left, n.Right)
		} else {
			fmt.Fprintf(s, "%v = %v", n.Left, n.Right)
		}

	case OASOP:
		if n.Implicit {
			if Op(n.Etype) == OADD {
				fmt.Fprintf(s, "%v++", n.Left)
			} else {
				fmt.Fprintf(s, "%v--", n.Left)
			}
			break
		}

		fmt.Fprintf(s, "%v %#v= %v", n.Left, Op(n.Etype), n.Right)

	case OAS2:
		if n.Colas && !complexinit {
			fmt.Fprintf(s, "%.v := %.v", n.List, n.Rlist)
			break
		}
		fallthrough

	case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		fmt.Fprintf(s, "%.v = %.v", n.List, n.Rlist)

	case ORETURN:
		fmt.Fprintf(s, "return %.v", n.List)

	case ORETJMP:
		fmt.Fprintf(s, "retjmp %v", n.Sym)

	case OPROC:
		fmt.Fprintf(s, "go %v", n.Left)

	case ODEFER:
		fmt.Fprintf(s, "defer %v", n.Left)

	case OIF:
		if simpleinit {
			fmt.Fprintf(s, "if %v; %v { %v }", n.Ninit.First(), n.Left, n.Nbody)
		} else {
			fmt.Fprintf(s, "if %v { %v }", n.Left, n.Nbody)
		}
		if n.Rlist.Len() != 0 {
			fmt.Fprintf(s, " else { %v }", n.Rlist)
		}

	case OFOR:
		if fmtmode == FErr { // TODO maybe only if FmtShort, same below
			fmt.Fprint(s, "for loop")
			break
		}

		fmt.Fprint(s, "for")
		if simpleinit {
			fmt.Fprintf(s, " %v;", n.Ninit.First())
		} else if n.Right != nil {
			fmt.Fprint(s, " ;")
		}

		if n.Left != nil {
			fmt.Fprintf(s, " %v", n.Left)
		}

		if n.Right != nil {
			fmt.Fprintf(s, "; %v", n.Right)
		} else if simpleinit {
			fmt.Fprint(s, ";")
		}

		fmt.Fprintf(s, " { %v }", n.Nbody)

	case ORANGE:
		if fmtmode == FErr {
			fmt.Fprint(s, "for loop")
			break
		}

		if n.List.Len() == 0 {
			fmt.Fprintf(s, "for range %v { %v }", n.Right, n.Nbody)
			break
		}

		fmt.Fprintf(s, "for %.v = range %v { %v }", n.List, n.Right, n.Nbody)

	case OSELECT, OSWITCH:
		if fmtmode == FErr {
			fmt.Fprintf(s, "%v statement", n.Op)
			break
		}

		fmt.Fprint(s, n.Op.GoString()) // %#v
		if simpleinit {
			fmt.Fprintf(s, " %v;", n.Ninit.First())
		}
		if n.Left != nil {
			fmt.Fprintf(s, " %v ", n.Left)
		}

		fmt.Fprintf(s, " { %v }", n.List)

	case OXCASE:
		if n.List.Len() != 0 {
			fmt.Fprintf(s, "case %.v", n.List)
		} else {
			fmt.Fprint(s, "default")
		}
		fmt.Fprintf(s, ": %v", n.Nbody)

	case OCASE:
		switch {
		case n.Left != nil:
			// single element
			fmt.Fprintf(s, "case %v", n.Left)
		case n.List.Len() > 0:
			// range
			if n.List.Len() != 2 {
				Fatalf("bad OCASE list length %d", n.List.Len())
			}
			fmt.Fprintf(s, "case %v..%v", n.List.First(), n.List.Second())
		default:
			fmt.Fprint(s, "default")
		}
		fmt.Fprintf(s, ": %v", n.Nbody)

	case OBREAK,
		OCONTINUE,
		OGOTO,
		OFALL,
		OXFALL:
		if n.Left != nil {
			fmt.Fprintf(s, "%#v %v", n.Op, n.Left)
		} else {
			fmt.Fprint(s, n.Op.GoString()) // %#v
		}

	case OEMPTY:
		break

	case OLABEL:
		fmt.Fprintf(s, "%v: ", n.Left)
	}

	if extrablock {
		fmt.Fprint(s, "}")
	}
}

var opprec = []int{
	OALIGNOF:      8,
	OAPPEND:       8,
	OARRAYBYTESTR: 8,
	OARRAYLIT:     8,
	OSLICELIT:     8,
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
	OOFFSETOF:     8,
	OPACK:         8,
	OPANIC:        8,
	OPAREN:        8,
	OPRINTN:       8,
	OPRINT:        8,
	ORUNESTR:      8,
	OSIZEOF:       8,
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

func (n *Node) exprfmt(s fmt.State, prec int) {
	for n != nil && n.Implicit && (n.Op == OIND || n.Op == OADDR) {
		n = n.Left
	}

	if n == nil {
		fmt.Fprint(s, "<N>")
		return
	}

	nprec := opprec[n.Op]
	if n.Op == OTYPE && n.Sym != nil {
		nprec = 8
	}

	if prec > nprec {
		fmt.Fprintf(s, "(%v)", n)
		return
	}

	switch n.Op {
	case OPAREN:
		fmt.Fprintf(s, "(%v)", n.Left)

	case ODDDARG:
		fmt.Fprint(s, "... argument")

	case OLITERAL: // this is a bit of a mess
		if fmtmode == FErr {
			if n.Orig != nil && n.Orig != n {
				n.Orig.exprfmt(s, prec)
				return
			}
			if n.Sym != nil {
				fmt.Fprint(s, n.Sym.String())
				return
			}
		}
		if n.Val().Ctype() == CTNIL && n.Orig != nil && n.Orig != n {
			n.Orig.exprfmt(s, prec)
			return
		}
		if n.Type != nil && n.Type.Etype != TIDEAL && n.Type.Etype != TNIL && n.Type != idealbool && n.Type != idealstring {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if n.Type.IsPtr() || (n.Type.IsChan() && n.Type.ChanDir() == Crecv) {
				fmt.Fprintf(s, "(%v)(%v)", n.Type, n.Val())
				return
			} else {
				fmt.Fprintf(s, "%v(%v)", n.Type, n.Val())
				return
			}
		}

		fmt.Fprintf(s, "%v", n.Val())

	// Special case: name used as local variable in export.
	// _ becomes ~b%d internally; print as _ for export
	case ONAME:
		if fmtmode == FErr && n.Sym != nil && n.Sym.Name[0] == '~' && n.Sym.Name[1] == 'b' {
			fmt.Fprint(s, "_")
			return
		}
		fallthrough
	case OPACK, ONONAME:
		fmt.Fprint(s, n.Sym.String())

	case OTYPE:
		if n.Type == nil && n.Sym != nil {
			fmt.Fprint(s, n.Sym.String())
			return
		}
		fmt.Fprintf(s, "%v", n.Type)

	case OTARRAY:
		if n.Left != nil {
			fmt.Fprintf(s, "[]%v", n.Left)
			return
		}
		fmt.Fprintf(s, "[]%v", n.Right) // happens before typecheck

	case OTMAP:
		fmt.Fprintf(s, "map[%v]%v", n.Left, n.Right)

	case OTCHAN:
		switch ChanDir(n.Etype) {
		case Crecv:
			fmt.Fprintf(s, "<-chan %v", n.Left)

		case Csend:
			fmt.Fprintf(s, "chan<- %v", n.Left)

		default:
			if n.Left != nil && n.Left.Op == OTCHAN && n.Left.Sym == nil && ChanDir(n.Left.Etype) == Crecv {
				fmt.Fprintf(s, "chan (%v)", n.Left)
			} else {
				fmt.Fprintf(s, "chan %v", n.Left)
			}
		}

	case OTSTRUCT:
		fmt.Fprint(s, "<struct>")

	case OTINTER:
		fmt.Fprint(s, "<inter>")

	case OTFUNC:
		fmt.Fprint(s, "<func>")

	case OCLOSURE:
		if fmtmode == FErr {
			fmt.Fprint(s, "func literal")
			return
		}
		if n.Nbody.Len() != 0 {
			fmt.Fprintf(s, "%v { %v }", n.Type, n.Nbody)
			return
		}
		fmt.Fprintf(s, "%v { %v }", n.Type, n.Func.Closure.Nbody)

	case OCOMPLIT:
		ptrlit := n.Right != nil && n.Right.Implicit && n.Right.Type != nil && n.Right.Type.IsPtr()
		if fmtmode == FErr {
			if n.Right != nil && n.Right.Type != nil && !n.Implicit {
				if ptrlit {
					fmt.Fprintf(s, "&%v literal", n.Right.Type.Elem())
					return
				} else {
					fmt.Fprintf(s, "%v literal", n.Right.Type)
					return
				}
			}

			fmt.Fprint(s, "composite literal")
			return
		}
		fmt.Fprintf(s, "(%v{ %.v })", n.Right, n.List)

	case OPTRLIT:
		fmt.Fprintf(s, "&%v", n.Left)

	case OSTRUCTLIT, OARRAYLIT, OSLICELIT, OMAPLIT:
		if fmtmode == FErr {
			fmt.Fprintf(s, "%v literal", n.Type)
			return
		}
		fmt.Fprintf(s, "(%v{ %.v })", n.Type, n.List)

	case OKEY:
		if n.Left != nil && n.Right != nil {
			fmt.Fprintf(s, "%v:%v", n.Left, n.Right)
			return
		}

		if n.Left == nil && n.Right != nil {
			fmt.Fprintf(s, ":%v", n.Right)
			return
		}
		if n.Left != nil && n.Right == nil {
			fmt.Fprintf(s, "%v:", n.Left)
			return
		}
		fmt.Fprint(s, ":")

	case OSTRUCTKEY:
		fmt.Fprintf(s, "%v:%v", n.Sym, n.Left)

	case OCALLPART:
		n.Left.exprfmt(s, nprec)
		if n.Right == nil || n.Right.Sym == nil {
			fmt.Fprint(s, ".<nil>")
			return
		}
		fmt.Fprintf(s, ".%0S", n.Right.Sym)

	case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
		n.Left.exprfmt(s, nprec)
		if n.Sym == nil {
			fmt.Fprint(s, ".<nil>")
			return
		}
		fmt.Fprintf(s, ".%0S", n.Sym)

	case ODOTTYPE, ODOTTYPE2:
		n.Left.exprfmt(s, nprec)
		if n.Right != nil {
			fmt.Fprintf(s, ".(%v)", n.Right)
			return
		}
		fmt.Fprintf(s, ".(%v)", n.Type)

	case OINDEX, OINDEXMAP:
		n.Left.exprfmt(s, nprec)
		fmt.Fprintf(s, "[%v]", n.Right)

	case OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
		n.Left.exprfmt(s, nprec)
		fmt.Fprint(s, "[")
		low, high, max := n.SliceBounds()
		if low != nil {
			fmt.Fprint(s, low.String())
		}
		fmt.Fprint(s, ":")
		if high != nil {
			fmt.Fprint(s, high.String())
		}
		if n.Op.IsSlice3() {
			fmt.Fprint(s, ":")
			if max != nil {
				fmt.Fprint(s, max.String())
			}
		}
		fmt.Fprint(s, "]")

	case OCOPY, OCOMPLEX:
		fmt.Fprintf(s, "%#v(%v, %v)", n.Op, n.Left, n.Right)

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		ORUNESTR:
		if n.Type == nil || n.Type.Sym == nil {
			fmt.Fprintf(s, "(%v)(%v)", n.Type, n.Left)
			return
		}
		if n.Left != nil {
			fmt.Fprintf(s, "%v(%v)", n.Type, n.Left)
			return
		}
		fmt.Fprintf(s, "%v(%.v)", n.Type, n.List)

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
		OALIGNOF,
		OOFFSETOF,
		OSIZEOF,
		OPRINT,
		OPRINTN:
		if n.Left != nil {
			fmt.Fprintf(s, "%#v(%v)", n.Op, n.Left)
			return
		}
		if n.Isddd {
			fmt.Fprintf(s, "%#v(%.v...)", n.Op, n.List)
			return
		}
		fmt.Fprintf(s, "%#v(%.v)", n.Op, n.List)

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		n.Left.exprfmt(s, nprec)
		if n.Isddd {
			fmt.Fprintf(s, "(%.v...)", n.List)
			return
		}
		fmt.Fprintf(s, "(%.v)", n.List)

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		if n.List.Len() != 0 { // pre-typecheck
			fmt.Fprintf(s, "make(%v, %.v)", n.Type, n.List)
			return
		}
		if n.Right != nil {
			fmt.Fprintf(s, "make(%v, %v, %v)", n.Type, n.Left, n.Right)
			return
		}
		if n.Left != nil && (n.Op == OMAKESLICE || !n.Left.Type.IsUntyped()) {
			fmt.Fprintf(s, "make(%v, %v)", n.Type, n.Left)
			return
		}
		fmt.Fprintf(s, "make(%v)", n.Type)

		// Unary
	case OPLUS,
		OMINUS,
		OADDR,
		OCOM,
		OIND,
		ONOT,
		ORECV:
		fmt.Fprint(s, n.Op.GoString()) // %#v
		if n.Left.Op == n.Op {
			fmt.Fprint(s, " ")
		}
		n.Left.exprfmt(s, nprec+1)

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
		n.Left.exprfmt(s, nprec)
		fmt.Fprintf(s, " %#v ", n.Op)
		n.Right.exprfmt(s, nprec+1)

	case OADDSTR:
		i := 0
		for _, n1 := range n.List.Slice() {
			if i != 0 {
				fmt.Fprint(s, " + ")
			}
			n1.exprfmt(s, nprec)
			i++
		}

	case OCMPSTR, OCMPIFACE:
		n.Left.exprfmt(s, nprec)
		// TODO(marvin): Fix Node.EType type union.
		fmt.Fprintf(s, " %#v ", Op(n.Etype))
		n.Right.exprfmt(s, nprec+1)

	default:
		fmt.Fprintf(s, "<node %v>", n.Op)
	}
}

func (n *Node) nodefmt(s fmt.State, flag FmtFlag) {
	t := n.Type

	// we almost always want the original, except in export mode for literals
	// this saves the importer some work, and avoids us having to redo some
	// special casing for package unsafe
	if n.Op != OLITERAL && n.Orig != nil {
		n = n.Orig
	}

	if flag&FmtLong != 0 && t != nil {
		if t.Etype == TNIL {
			fmt.Fprint(s, "nil")
		} else {
			fmt.Fprintf(s, "%v (type %v)", n, t)
		}
		return
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if opprec[n.Op] < 0 {
		n.stmtfmt(s)
		return
	}

	n.exprfmt(s, 0)
}

func (n *Node) nodedump(s fmt.State, flag FmtFlag) {
	if n == nil {
		return
	}

	recur := flag&FmtShort == 0

	if recur {
		indent(s)
		if dumpdepth > 10 {
			fmt.Fprint(s, "...")
			return
		}

		if n.Ninit.Len() != 0 {
			fmt.Fprintf(s, "%v-init%v", n.Op, n.Ninit)
			indent(s)
		}
	}

	switch n.Op {
	default:
		fmt.Fprintf(s, "%v%j", n.Op, n)

	case OINDREGSP:
		fmt.Fprintf(s, "%v-SP%j", n.Op, n)

	case OLITERAL:
		fmt.Fprintf(s, "%v-%v%j", n.Op, n.Val(), n)

	case ONAME, ONONAME:
		if n.Sym != nil {
			fmt.Fprintf(s, "%v-%v%j", n.Op, n.Sym, n)
		} else {
			fmt.Fprintf(s, "%v%j", n.Op, n)
		}
		if recur && n.Type == nil && n.Name != nil && n.Name.Param != nil && n.Name.Param.Ntype != nil {
			indent(s)
			fmt.Fprintf(s, "%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}

	case OASOP:
		fmt.Fprintf(s, "%v-%v%j", n.Op, Op(n.Etype), n)

	case OTYPE:
		fmt.Fprintf(s, "%v %v%j type=%v", n.Op, n.Sym, n, n.Type)
		if recur && n.Type == nil && n.Name.Param.Ntype != nil {
			indent(s)
			fmt.Fprintf(s, "%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}
	}

	if n.Sym != nil && n.Op != ONAME {
		fmt.Fprintf(s, " %v", n.Sym)
	}

	if n.Type != nil {
		fmt.Fprintf(s, " %v", n.Type)
	}

	if recur {
		if n.Left != nil {
			fmt.Fprintf(s, "%v", n.Left)
		}
		if n.Right != nil {
			fmt.Fprintf(s, "%v", n.Right)
		}
		if n.List.Len() != 0 {
			indent(s)
			fmt.Fprintf(s, "%v-list%v", n.Op, n.List)
		}

		if n.Rlist.Len() != 0 {
			indent(s)
			fmt.Fprintf(s, "%v-rlist%v", n.Op, n.Rlist)
		}

		if n.Nbody.Len() != 0 {
			indent(s)
			fmt.Fprintf(s, "%v-body%v", n.Op, n.Nbody)
		}
	}
}

// "%S" suppresses qualifying with package
func (s *Sym) Format(f fmt.State, verb rune) {
	switch verb {
	case 'v', 'S':
		fmt.Fprint(f, s.sconv(fmtFlag(f, verb)))

	default:
		fmt.Fprintf(f, "%%!%c(*Sym=%p)", verb, s)
	}
}

func (s *Sym) String() string {
	return s.sconv(0)
}

// See #16897 before changing the implementation of sconv.
func (s *Sym) sconv(flag FmtFlag) string {
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
	sm := setfmode(&flag)
	str := s.symfmt(flag)
	flag = sf
	fmtmode = sm
	return str
}

func (t *Type) String() string {
	return t.tconv(0)
}

func fldconv(f *Field, flag FmtFlag) string {
	if f == nil {
		return "<T>"
	}

	sf := flag
	sm := setfmode(&flag)

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
						s = lookup("_")
					}
				}
			} else {
				s = nil
			}
		}

		if s != nil && f.Embedded == 0 {
			if f.Funarg != FunargNone {
				name = f.Nname.String()
			} else if flag&FmtLong != 0 {
				name = fmt.Sprintf("%0S", s)
				if !exportname(name) && flag&FmtUnsigned == 0 {
					name = s.String() // qualify non-exported names (used on structs, not on funarg)
				}
			} else {
				name = s.String()
			}
		}
	}

	var typ string
	if f.Isddd {
		typ = fmt.Sprintf("...%v", f.Type.Elem())
	} else {
		typ = fmt.Sprintf("%v", f.Type)
	}

	str := typ
	if name != "" {
		str = name + " " + typ
	}

	if flag&FmtShort == 0 && f.Funarg == FunargNone && f.Note != "" {
		str += " " + strconv.Quote(f.Note)
	}

	if fmtmode == FTypeId && (sf&FmtUnsigned != 0) {
		fmtpkgpfx--
	}

	flag = sf
	fmtmode = sm
	return str
}

// "%L"  print definition, not name
// "%S"  omit 'func' and receiver from function types, short type names
// "% v" package name, not prefix (FTypeId mode, sticky)
func (t *Type) Format(s fmt.State, verb rune) {
	switch verb {
	case 'v', 'S', 'L':
		fmt.Fprint(s, t.tconv(fmtFlag(s, verb)))

	default:
		fmt.Fprintf(s, "%%!%c(*Type=%p)", verb, t)
	}
}

// See #16897 before changing the implementation of tconv.
func (t *Type) tconv(flag FmtFlag) string {
	if t == nil {
		return "<T>"
	}

	if t.Trecur > 4 {
		return "<...>"
	}

	t.Trecur++
	sf := flag
	sm := setfmode(&flag)

	if fmtmode == FTypeId && (sf&FmtUnsigned != 0) {
		fmtpkgpfx++
	}
	if fmtpkgpfx != 0 {
		flag |= FmtUnsigned
	}

	str := t.typefmt(flag)

	if fmtmode == FTypeId && (sf&FmtUnsigned != 0) {
		fmtpkgpfx--
	}

	flag = sf
	fmtmode = sm
	t.Trecur--
	return str
}

func (n *Node) String() string {
	return fmt.Sprint(n)
}

// "%L"  suffix with "(type %T)" where possible
// "%+S" in debug mode, don't recurse, no multiline output
func (n *Node) Nconv(s fmt.State, flag FmtFlag) {
	if n == nil {
		fmt.Fprint(s, "<N>")
		return
	}

	sf := flag
	sm := setfmode(&flag)

	switch fmtmode {
	case FErr:
		n.nodefmt(s, flag)

	case FDbg:
		dumpdepth++
		n.nodedump(s, flag)
		dumpdepth--

	default:
		Fatalf("unhandled %%N mode: %d", fmtmode)
	}

	flag = sf
	fmtmode = sm
}

func (l Nodes) Format(s fmt.State, verb rune) {
	switch verb {
	case 'v':
		l.hconv(s, fmtFlag(s, verb))

	default:
		fmt.Fprintf(s, "%%!%c(Nodes)", verb)
	}
}

func (n Nodes) String() string {
	return fmt.Sprint(n)
}

// Flags: all those of %N plus '.': separate with comma's instead of semicolons.
func (l Nodes) hconv(s fmt.State, flag FmtFlag) {
	if l.Len() == 0 && fmtmode == FDbg {
		fmt.Fprint(s, "<nil>")
		return
	}

	sf := flag
	sm := setfmode(&flag)
	sep := "; "
	if fmtmode == FDbg {
		sep = "\n"
	} else if flag&FmtComma != 0 {
		sep = ", "
	}

	for i, n := range l.Slice() {
		fmt.Fprint(s, n)
		if i+1 < l.Len() {
			fmt.Fprint(s, sep)
		}
	}

	flag = sf
	fmtmode = sm
}

func dumplist(s string, l Nodes) {
	fmt.Printf("%s%+v\n", s, l)
}

func Dump(s string, n *Node) {
	fmt.Printf("%s [%p]%+v\n", s, n, n)
}

// TODO(gri) make variable local somehow
var dumpdepth int

// indent prints indentation to s.
func indent(s fmt.State) {
	fmt.Fprint(s, "\n")
	for i := 0; i < dumpdepth; i++ {
		fmt.Fprint(s, ".   ")
	}
}
