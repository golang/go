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

const ( //                                 fmt.Format flag/prec or verb
	FmtLeft     FmtFlag = 1 << iota // '-'
	FmtSharp                        // '#'
	FmtSign                         // '+'
	FmtUnsigned                     // internal use only (historic: u flag)
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
		Fatalf("FmtUnsigned in format string")
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
//		Flags:  #: print Go syntax (automatic unless mode == FDbg)
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
	FTypeIdName // same as FTypeId, but use package name instead of prefix
)

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

// update returns the results of applying f to mode.
func (f FmtFlag) update(mode fmtMode) (FmtFlag, fmtMode) {
	switch {
	case f&FmtSign != 0:
		mode = FDbg
	case f&FmtSharp != 0:
		// ignore (textual export format no longer supported)
	case f&FmtUnsigned != 0:
		mode = FTypeIdName
	case f&FmtLeft != 0:
		mode = FTypeId
	}

	f &^= FmtSharp | FmtLeft | FmtSign
	return f, mode
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
	ODELETE:   "delete",
	ODEFER:    "defer",
	ODIV:      "/",
	OEQ:       "==",
	OFALL:     "fallthrough",
	OFOR:      "for",
	OFORUNTIL: "foruntil", // not actual syntax; used to avoid off-end pointer live on backedge.892
	OGE:       ">=",
	OGOTO:     "goto",
	OGT:       ">",
	OIF:       "if",
	OIMAG:     "imag",
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

func (o Op) format(s fmt.State, verb rune, mode fmtMode) {
	switch verb {
	case 'v':
		o.oconv(s, fmtFlag(s, verb), mode)

	default:
		fmt.Fprintf(s, "%%!%c(Op=%d)", verb, int(o))
	}
}

func (o Op) oconv(s fmt.State, flag FmtFlag, mode fmtMode) {
	if flag&FmtSharp != 0 || mode != FDbg {
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

type (
	fmtMode int

	fmtNodeErr        Node
	fmtNodeDbg        Node
	fmtNodeTypeId     Node
	fmtNodeTypeIdName Node

	fmtOpErr        Op
	fmtOpDbg        Op
	fmtOpTypeId     Op
	fmtOpTypeIdName Op

	fmtTypeErr        Type
	fmtTypeDbg        Type
	fmtTypeTypeId     Type
	fmtTypeTypeIdName Type

	fmtSymErr        Sym
	fmtSymDbg        Sym
	fmtSymTypeId     Sym
	fmtSymTypeIdName Sym

	fmtNodesErr        Nodes
	fmtNodesDbg        Nodes
	fmtNodesTypeId     Nodes
	fmtNodesTypeIdName Nodes
)

func (n *fmtNodeErr) Format(s fmt.State, verb rune)        { (*Node)(n).format(s, verb, FErr) }
func (n *fmtNodeDbg) Format(s fmt.State, verb rune)        { (*Node)(n).format(s, verb, FDbg) }
func (n *fmtNodeTypeId) Format(s fmt.State, verb rune)     { (*Node)(n).format(s, verb, FTypeId) }
func (n *fmtNodeTypeIdName) Format(s fmt.State, verb rune) { (*Node)(n).format(s, verb, FTypeIdName) }
func (n *Node) Format(s fmt.State, verb rune)              { n.format(s, verb, FErr) }

func (o fmtOpErr) Format(s fmt.State, verb rune)        { Op(o).format(s, verb, FErr) }
func (o fmtOpDbg) Format(s fmt.State, verb rune)        { Op(o).format(s, verb, FDbg) }
func (o fmtOpTypeId) Format(s fmt.State, verb rune)     { Op(o).format(s, verb, FTypeId) }
func (o fmtOpTypeIdName) Format(s fmt.State, verb rune) { Op(o).format(s, verb, FTypeIdName) }
func (o Op) Format(s fmt.State, verb rune)              { o.format(s, verb, FErr) }

func (t *fmtTypeErr) Format(s fmt.State, verb rune)        { (*Type)(t).format(s, verb, FErr) }
func (t *fmtTypeDbg) Format(s fmt.State, verb rune)        { (*Type)(t).format(s, verb, FDbg) }
func (t *fmtTypeTypeId) Format(s fmt.State, verb rune)     { (*Type)(t).format(s, verb, FTypeId) }
func (t *fmtTypeTypeIdName) Format(s fmt.State, verb rune) { (*Type)(t).format(s, verb, FTypeIdName) }
func (t *Type) Format(s fmt.State, verb rune)              { t.format(s, verb, FErr) }

func (y *fmtSymErr) Format(s fmt.State, verb rune)        { (*Sym)(y).format(s, verb, FErr) }
func (y *fmtSymDbg) Format(s fmt.State, verb rune)        { (*Sym)(y).format(s, verb, FDbg) }
func (y *fmtSymTypeId) Format(s fmt.State, verb rune)     { (*Sym)(y).format(s, verb, FTypeId) }
func (y *fmtSymTypeIdName) Format(s fmt.State, verb rune) { (*Sym)(y).format(s, verb, FTypeIdName) }
func (y *Sym) Format(s fmt.State, verb rune)              { y.format(s, verb, FErr) }

func (n fmtNodesErr) Format(s fmt.State, verb rune)        { (Nodes)(n).format(s, verb, FErr) }
func (n fmtNodesDbg) Format(s fmt.State, verb rune)        { (Nodes)(n).format(s, verb, FDbg) }
func (n fmtNodesTypeId) Format(s fmt.State, verb rune)     { (Nodes)(n).format(s, verb, FTypeId) }
func (n fmtNodesTypeIdName) Format(s fmt.State, verb rune) { (Nodes)(n).format(s, verb, FTypeIdName) }
func (n Nodes) Format(s fmt.State, verb rune)              { n.format(s, verb, FErr) }

func (m fmtMode) Fprintf(s fmt.State, format string, args ...interface{}) {
	m.prepareArgs(args)
	fmt.Fprintf(s, format, args...)
}

func (m fmtMode) Sprintf(format string, args ...interface{}) string {
	m.prepareArgs(args)
	return fmt.Sprintf(format, args...)
}

func (m fmtMode) Sprint(args ...interface{}) string {
	m.prepareArgs(args)
	return fmt.Sprint(args...)
}

func (m fmtMode) prepareArgs(args []interface{}) {
	switch m {
	case FErr:
		for i, arg := range args {
			switch arg := arg.(type) {
			case Op:
				args[i] = fmtOpErr(arg)
			case *Node:
				args[i] = (*fmtNodeErr)(arg)
			case *Type:
				args[i] = (*fmtTypeErr)(arg)
			case *Sym:
				args[i] = (*fmtSymErr)(arg)
			case Nodes:
				args[i] = fmtNodesErr(arg)
			case Val, int32, int64, string, EType:
				// OK: printing these types doesn't depend on mode
			default:
				Fatalf("mode.prepareArgs type %T", arg)
			}
		}
	case FDbg:
		for i, arg := range args {
			switch arg := arg.(type) {
			case Op:
				args[i] = fmtOpDbg(arg)
			case *Node:
				args[i] = (*fmtNodeDbg)(arg)
			case *Type:
				args[i] = (*fmtTypeDbg)(arg)
			case *Sym:
				args[i] = (*fmtSymDbg)(arg)
			case Nodes:
				args[i] = fmtNodesDbg(arg)
			case Val, int32, int64, string, EType:
				// OK: printing these types doesn't depend on mode
			default:
				Fatalf("mode.prepareArgs type %T", arg)
			}
		}
	case FTypeId:
		for i, arg := range args {
			switch arg := arg.(type) {
			case Op:
				args[i] = fmtOpTypeId(arg)
			case *Node:
				args[i] = (*fmtNodeTypeId)(arg)
			case *Type:
				args[i] = (*fmtTypeTypeId)(arg)
			case *Sym:
				args[i] = (*fmtSymTypeId)(arg)
			case Nodes:
				args[i] = fmtNodesTypeId(arg)
			case Val, int32, int64, string, EType:
				// OK: printing these types doesn't depend on mode
			default:
				Fatalf("mode.prepareArgs type %T", arg)
			}
		}
	case FTypeIdName:
		for i, arg := range args {
			switch arg := arg.(type) {
			case Op:
				args[i] = fmtOpTypeIdName(arg)
			case *Node:
				args[i] = (*fmtNodeTypeIdName)(arg)
			case *Type:
				args[i] = (*fmtTypeTypeIdName)(arg)
			case *Sym:
				args[i] = (*fmtSymTypeIdName)(arg)
			case Nodes:
				args[i] = fmtNodesTypeIdName(arg)
			case Val, int32, int64, string, EType:
				// OK: printing these types doesn't depend on mode
			default:
				Fatalf("mode.prepareArgs type %T", arg)
			}
		}
	default:
		Fatalf("mode.prepareArgs mode %d", m)
	}
}

func (n *Node) format(s fmt.State, verb rune, mode fmtMode) {
	switch verb {
	case 'v', 'S', 'L':
		n.nconv(s, fmtFlag(s, verb), mode)

	case 'j':
		n.jconv(s, fmtFlag(s, verb))

	default:
		fmt.Fprintf(s, "%%!%c(*Node=%p)", verb, n)
	}
}

// *Node details
func (n *Node) jconv(s fmt.State, flag FmtFlag) {
	c := flag & FmtShort

	if c == 0 && n.Addable() {
		fmt.Fprintf(s, " a(%v)", n.Addable())
	}

	if c == 0 && n.Name != nil && n.Name.Vargen != 0 {
		fmt.Fprintf(s, " g(%d)", n.Name.Vargen)
	}

	if n.Pos.IsKnown() {
		fmt.Fprintf(s, " l(%d)", n.Pos.Line())
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

	if n.Colas() {
		fmt.Fprintf(s, " colas(%v)", n.Colas())
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

	if n.Isddd() {
		fmt.Fprintf(s, " isddd(%v)", n.Isddd())
	}

	if n.Implicit() {
		fmt.Fprintf(s, " implicit(%v)", n.Implicit())
	}

	if n.Embedded != 0 {
		fmt.Fprintf(s, " embedded(%d)", n.Embedded)
	}

	if n.Addrtaken() {
		fmt.Fprint(s, " addrtaken")
	}

	if n.Assigned() {
		fmt.Fprint(s, " assigned")
	}
	if n.Bounded() {
		fmt.Fprint(s, " bounded")
	}
	if n.NonNil() {
		fmt.Fprint(s, " nonnil")
	}

	if c == 0 && n.HasCall() {
		fmt.Fprintf(s, " hascall")
	}

	if c == 0 && n.Used() {
		fmt.Fprintf(s, " used(%v)", n.Used())
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
		fmt.Fprint(s, u)

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
	TDDDFIELD:   "TDDDFIELD",
}

func (et EType) String() string {
	if int(et) < len(etnames) && etnames[et] != "" {
		return etnames[et]
	}
	return fmt.Sprintf("E-%d", et)
}

func (s *Sym) symfmt(flag FmtFlag, mode fmtMode) string {
	if s.Pkg != nil && flag&FmtShort == 0 {
		switch mode {
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

		case FTypeIdName:
			return s.Pkg.Name + "." + s.Name // dcommontype, typehash

		case FTypeId:
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

		if mode == FDbg {
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

func (t *Type) typefmt(flag FmtFlag, mode fmtMode, depth int) string {
	if t == nil {
		return "<T>"
	}

	if t == bytetype || t == runetype {
		// in %-T mode collapse rune and byte with their originals.
		switch mode {
		case FTypeIdName, FTypeId:
			t = Types[t.Etype]
		default:
			return t.Sym.sconv(FmtShort, mode)
		}
	}

	if t == errortype {
		return "error"
	}

	// Unless the 'l' flag was specified, if the type has a name, just print that name.
	if flag&FmtLong == 0 && t.Sym != nil && t != Types[t.Etype] {
		switch mode {
		case FTypeId, FTypeIdName:
			if flag&FmtShort != 0 {
				if t.Vargen != 0 {
					return mode.Sprintf("%v·%d", t.Sym.sconv(FmtShort, mode), t.Vargen)
				}
				return t.Sym.sconv(FmtShort, mode)
			}

			if mode == FTypeIdName {
				return t.Sym.sconv(FmtUnsigned, mode)
			}

			if t.Sym.Pkg == localpkg && t.Vargen != 0 {
				return mode.Sprintf("%v·%d", t.Sym, t.Vargen)
			}
		}

		return t.Sym.modeString(mode)
	}

	if int(t.Etype) < len(basicnames) && basicnames[t.Etype] != "" {
		prefix := ""
		if mode == FErr && (t == idealbool || t == idealstring) {
			prefix = "untyped "
		}
		return prefix + basicnames[t.Etype]
	}

	if mode == FDbg {
		return t.Etype.String() + "-" + t.typefmt(flag, 0, depth)
	}

	switch t.Etype {
	case TPTR32, TPTR64:
		switch mode {
		case FTypeId, FTypeIdName:
			if flag&FmtShort != 0 {
				return "*" + t.Elem().tconv(FmtShort, mode, depth)
			}
		}
		return "*" + t.Elem().modeString(mode, depth)

	case TARRAY:
		if t.isDDDArray() {
			return "[...]" + t.Elem().modeString(mode, depth)
		}
		return "[" + strconv.FormatInt(t.NumElem(), 10) + "]" + t.Elem().modeString(mode, depth)

	case TSLICE:
		return "[]" + t.Elem().modeString(mode, depth)

	case TCHAN:
		switch t.ChanDir() {
		case Crecv:
			return "<-chan " + t.Elem().modeString(mode, depth)

		case Csend:
			return "chan<- " + t.Elem().modeString(mode, depth)
		}

		if t.Elem() != nil && t.Elem().IsChan() && t.Elem().Sym == nil && t.Elem().ChanDir() == Crecv {
			return "chan (" + t.Elem().modeString(mode, depth) + ")"
		}
		return "chan " + t.Elem().modeString(mode, depth)

	case TMAP:
		return "map[" + t.Key().modeString(mode, depth) + "]" + t.Val().modeString(mode, depth)

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
				buf = append(buf, f.Sym.sconv(FmtShort, mode)...)
			default:
				buf = append(buf, f.Sym.sconv(FmtUnsigned, mode)...)
			}
			buf = append(buf, f.Type.tconv(FmtShort, mode, depth)...)
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
				buf = append(buf, t.Recvs().modeString(mode, depth)...)
				buf = append(buf, ' ')
			}
			buf = append(buf, "func"...)
		}
		buf = append(buf, t.Params().modeString(mode, depth)...)

		switch t.Results().NumFields() {
		case 0:
			// nothing to do

		case 1:
			buf = append(buf, ' ')
			buf = append(buf, t.Results().Field(0).Type.modeString(mode, depth)...) // struct->field->field's type

		default:
			buf = append(buf, ' ')
			buf = append(buf, t.Results().modeString(mode, depth)...)
		}
		return string(buf)

	case TSTRUCT:
		if m := t.StructType().Map; m != nil {
			mt := m.MapType()
			// Format the bucket struct for map[x]y as map.bucket[x]y.
			// This avoids a recursive print that generates very long names.
			if mt.Bucket == t {
				return "map.bucket[" + m.Key().modeString(mode, depth) + "]" + m.Val().modeString(mode, depth)
			}

			if mt.Hmap == t {
				return "map.hdr[" + m.Key().modeString(mode, depth) + "]" + m.Val().modeString(mode, depth)
			}

			if mt.Hiter == t {
				return "map.iter[" + m.Key().modeString(mode, depth) + "]" + m.Val().modeString(mode, depth)
			}

			Fatalf("unknown internal map type")
		}

		buf := make([]byte, 0, 64)
		if t.IsFuncArgStruct() {
			buf = append(buf, '(')
			var flag1 FmtFlag
			switch mode {
			case FTypeId, FTypeIdName, FErr:
				// no argument names on function signature, and no "noescape"/"nosplit" tags
				flag1 = FmtShort
			}
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					buf = append(buf, ", "...)
				}
				buf = append(buf, fldconv(f, flag1, mode, depth)...)
			}
			buf = append(buf, ')')
		} else {
			buf = append(buf, "struct {"...)
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					buf = append(buf, ';')
				}
				buf = append(buf, ' ')
				buf = append(buf, fldconv(f, FmtLong, mode, depth)...)
			}
			if t.NumFields() != 0 {
				buf = append(buf, ' ')
			}
			buf = append(buf, '}')
		}
		return string(buf)

	case TFORW:
		if t.Sym != nil {
			return "undefined " + t.Sym.modeString(mode)
		}
		return "undefined"

	case TUNSAFEPTR:
		return "unsafe.Pointer"

	case TDDDFIELD:
		return mode.Sprintf("%v <%v> %v", t.Etype, t.Sym, t.DDDField())

	case Txxx:
		return "Txxx"
	}

	// Don't know how to handle - fall back to detailed prints.
	return mode.Sprintf("%v <%v>", t.Etype, t.Sym)
}

// Statements which may be rendered with a simplestmt as init.
func stmtwithinit(op Op) bool {
	switch op {
	case OIF, OFOR, OFORUNTIL, OSWITCH:
		return true
	}

	return false
}

func (n *Node) stmtfmt(s fmt.State, mode fmtMode) {
	// some statements allow for an init, but at most one,
	// but we may have an arbitrary number added, eg by typecheck
	// and inlining. If it doesn't fit the syntax, emit an enclosing
	// block starting with the init statements.

	// if we can just say "for" n->ninit; ... then do so
	simpleinit := n.Ninit.Len() == 1 && n.Ninit.First().Ninit.Len() == 0 && stmtwithinit(n.Op)

	// otherwise, print the inits as separate statements
	complexinit := n.Ninit.Len() != 0 && !simpleinit && (mode != FErr)

	// but if it was for if/for/switch, put in an extra surrounding block to limit the scope
	extrablock := complexinit && stmtwithinit(n.Op)

	if extrablock {
		fmt.Fprint(s, "{")
	}

	if complexinit {
		mode.Fprintf(s, " %v; ", n.Ninit)
	}

	switch n.Op {
	case ODCL:
		mode.Fprintf(s, "var %v %v", n.Left.Sym, n.Left.Type)

	case ODCLFIELD:
		if n.Left != nil {
			mode.Fprintf(s, "%v %v", n.Left, n.Right)
		} else {
			mode.Fprintf(s, "%v", n.Right)
		}

	// Don't export "v = <N>" initializing statements, hope they're always
	// preceded by the DCL which will be re-parsed and typechecked to reproduce
	// the "v = <N>" again.
	case OAS:
		if n.Colas() && !complexinit {
			mode.Fprintf(s, "%v := %v", n.Left, n.Right)
		} else {
			mode.Fprintf(s, "%v = %v", n.Left, n.Right)
		}

	case OASOP:
		if n.Implicit() {
			if Op(n.Etype) == OADD {
				mode.Fprintf(s, "%v++", n.Left)
			} else {
				mode.Fprintf(s, "%v--", n.Left)
			}
			break
		}

		mode.Fprintf(s, "%v %#v= %v", n.Left, Op(n.Etype), n.Right)

	case OAS2:
		if n.Colas() && !complexinit {
			mode.Fprintf(s, "%.v := %.v", n.List, n.Rlist)
			break
		}
		fallthrough

	case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		mode.Fprintf(s, "%.v = %.v", n.List, n.Rlist)

	case ORETURN:
		mode.Fprintf(s, "return %.v", n.List)

	case ORETJMP:
		mode.Fprintf(s, "retjmp %v", n.Sym)

	case OPROC:
		mode.Fprintf(s, "go %v", n.Left)

	case ODEFER:
		mode.Fprintf(s, "defer %v", n.Left)

	case OIF:
		if simpleinit {
			mode.Fprintf(s, "if %v; %v { %v }", n.Ninit.First(), n.Left, n.Nbody)
		} else {
			mode.Fprintf(s, "if %v { %v }", n.Left, n.Nbody)
		}
		if n.Rlist.Len() != 0 {
			mode.Fprintf(s, " else { %v }", n.Rlist)
		}

	case OFOR, OFORUNTIL:
		opname := "for"
		if n.Op == OFORUNTIL {
			opname = "foruntil"
		}
		if mode == FErr { // TODO maybe only if FmtShort, same below
			fmt.Fprintf(s, "%s loop", opname)
			break
		}

		fmt.Fprint(s, opname)
		if simpleinit {
			mode.Fprintf(s, " %v;", n.Ninit.First())
		} else if n.Right != nil {
			fmt.Fprint(s, " ;")
		}

		if n.Left != nil {
			mode.Fprintf(s, " %v", n.Left)
		}

		if n.Right != nil {
			mode.Fprintf(s, "; %v", n.Right)
		} else if simpleinit {
			fmt.Fprint(s, ";")
		}

		mode.Fprintf(s, " { %v }", n.Nbody)

	case ORANGE:
		if mode == FErr {
			fmt.Fprint(s, "for loop")
			break
		}

		if n.List.Len() == 0 {
			mode.Fprintf(s, "for range %v { %v }", n.Right, n.Nbody)
			break
		}

		mode.Fprintf(s, "for %.v = range %v { %v }", n.List, n.Right, n.Nbody)

	case OSELECT, OSWITCH:
		if mode == FErr {
			mode.Fprintf(s, "%v statement", n.Op)
			break
		}

		mode.Fprintf(s, "%#v", n.Op)
		if simpleinit {
			mode.Fprintf(s, " %v;", n.Ninit.First())
		}
		if n.Left != nil {
			mode.Fprintf(s, " %v ", n.Left)
		}

		mode.Fprintf(s, " { %v }", n.List)

	case OXCASE:
		if n.List.Len() != 0 {
			mode.Fprintf(s, "case %.v", n.List)
		} else {
			fmt.Fprint(s, "default")
		}
		mode.Fprintf(s, ": %v", n.Nbody)

	case OCASE:
		switch {
		case n.Left != nil:
			// single element
			mode.Fprintf(s, "case %v", n.Left)
		case n.List.Len() > 0:
			// range
			if n.List.Len() != 2 {
				Fatalf("bad OCASE list length %d", n.List.Len())
			}
			mode.Fprintf(s, "case %v..%v", n.List.First(), n.List.Second())
		default:
			fmt.Fprint(s, "default")
		}
		mode.Fprintf(s, ": %v", n.Nbody)

	case OBREAK,
		OCONTINUE,
		OGOTO,
		OFALL,
		OXFALL:
		if n.Left != nil {
			mode.Fprintf(s, "%#v %v", n.Op, n.Left)
		} else {
			mode.Fprintf(s, "%#v", n.Op)
		}

	case OEMPTY:
		break

	case OLABEL:
		mode.Fprintf(s, "%v: ", n.Left)
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
	OFORUNTIL:   -1,
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

	OEND: 0,
}

func (n *Node) exprfmt(s fmt.State, prec int, mode fmtMode) {
	for n != nil && n.Implicit() && (n.Op == OIND || n.Op == OADDR) {
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
		mode.Fprintf(s, "(%v)", n)
		return
	}

	switch n.Op {
	case OPAREN:
		mode.Fprintf(s, "(%v)", n.Left)

	case ODDDARG:
		fmt.Fprint(s, "... argument")

	case OLITERAL: // this is a bit of a mess
		if mode == FErr {
			if n.Orig != nil && n.Orig != n {
				n.Orig.exprfmt(s, prec, mode)
				return
			}
			if n.Sym != nil {
				fmt.Fprint(s, n.Sym.modeString(mode))
				return
			}
		}
		if n.Val().Ctype() == CTNIL && n.Orig != nil && n.Orig != n {
			n.Orig.exprfmt(s, prec, mode)
			return
		}
		if n.Type != nil && n.Type.Etype != TIDEAL && n.Type.Etype != TNIL && n.Type != idealbool && n.Type != idealstring {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if n.Type.IsPtr() || (n.Type.IsChan() && n.Type.ChanDir() == Crecv) {
				mode.Fprintf(s, "(%v)(%v)", n.Type, n.Val())
				return
			} else {
				mode.Fprintf(s, "%v(%v)", n.Type, n.Val())
				return
			}
		}

		mode.Fprintf(s, "%v", n.Val())

	// Special case: name used as local variable in export.
	// _ becomes ~b%d internally; print as _ for export
	case ONAME:
		if mode == FErr && n.Sym != nil && n.Sym.Name[0] == '~' && n.Sym.Name[1] == 'b' {
			fmt.Fprint(s, "_")
			return
		}
		fallthrough
	case OPACK, ONONAME:
		fmt.Fprint(s, n.Sym.modeString(mode))

	case OTYPE:
		if n.Type == nil && n.Sym != nil {
			fmt.Fprint(s, n.Sym.modeString(mode))
			return
		}
		mode.Fprintf(s, "%v", n.Type)

	case OTARRAY:
		if n.Left != nil {
			mode.Fprintf(s, "[]%v", n.Left)
			return
		}
		mode.Fprintf(s, "[]%v", n.Right) // happens before typecheck

	case OTMAP:
		mode.Fprintf(s, "map[%v]%v", n.Left, n.Right)

	case OTCHAN:
		switch ChanDir(n.Etype) {
		case Crecv:
			mode.Fprintf(s, "<-chan %v", n.Left)

		case Csend:
			mode.Fprintf(s, "chan<- %v", n.Left)

		default:
			if n.Left != nil && n.Left.Op == OTCHAN && n.Left.Sym == nil && ChanDir(n.Left.Etype) == Crecv {
				mode.Fprintf(s, "chan (%v)", n.Left)
			} else {
				mode.Fprintf(s, "chan %v", n.Left)
			}
		}

	case OTSTRUCT:
		fmt.Fprint(s, "<struct>")

	case OTINTER:
		fmt.Fprint(s, "<inter>")

	case OTFUNC:
		fmt.Fprint(s, "<func>")

	case OCLOSURE:
		if mode == FErr {
			fmt.Fprint(s, "func literal")
			return
		}
		if n.Nbody.Len() != 0 {
			mode.Fprintf(s, "%v { %v }", n.Type, n.Nbody)
			return
		}
		mode.Fprintf(s, "%v { %v }", n.Type, n.Func.Closure.Nbody)

	case OCOMPLIT:
		ptrlit := n.Right != nil && n.Right.Implicit() && n.Right.Type != nil && n.Right.Type.IsPtr()
		if mode == FErr {
			if n.Right != nil && n.Right.Type != nil && !n.Implicit() {
				if ptrlit {
					mode.Fprintf(s, "&%v literal", n.Right.Type.Elem())
					return
				} else {
					mode.Fprintf(s, "%v literal", n.Right.Type)
					return
				}
			}

			fmt.Fprint(s, "composite literal")
			return
		}
		mode.Fprintf(s, "(%v{ %.v })", n.Right, n.List)

	case OPTRLIT:
		mode.Fprintf(s, "&%v", n.Left)

	case OSTRUCTLIT, OARRAYLIT, OSLICELIT, OMAPLIT:
		if mode == FErr {
			mode.Fprintf(s, "%v literal", n.Type)
			return
		}
		mode.Fprintf(s, "(%v{ %.v })", n.Type, n.List)

	case OKEY:
		if n.Left != nil && n.Right != nil {
			mode.Fprintf(s, "%v:%v", n.Left, n.Right)
			return
		}

		if n.Left == nil && n.Right != nil {
			mode.Fprintf(s, ":%v", n.Right)
			return
		}
		if n.Left != nil && n.Right == nil {
			mode.Fprintf(s, "%v:", n.Left)
			return
		}
		fmt.Fprint(s, ":")

	case OSTRUCTKEY:
		mode.Fprintf(s, "%v:%v", n.Sym, n.Left)

	case OCALLPART:
		n.Left.exprfmt(s, nprec, mode)
		if n.Right == nil || n.Right.Sym == nil {
			fmt.Fprint(s, ".<nil>")
			return
		}
		mode.Fprintf(s, ".%0S", n.Right.Sym)

	case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
		n.Left.exprfmt(s, nprec, mode)
		if n.Sym == nil {
			fmt.Fprint(s, ".<nil>")
			return
		}
		mode.Fprintf(s, ".%0S", n.Sym)

	case ODOTTYPE, ODOTTYPE2:
		n.Left.exprfmt(s, nprec, mode)
		if n.Right != nil {
			mode.Fprintf(s, ".(%v)", n.Right)
			return
		}
		mode.Fprintf(s, ".(%v)", n.Type)

	case OINDEX, OINDEXMAP:
		n.Left.exprfmt(s, nprec, mode)
		mode.Fprintf(s, "[%v]", n.Right)

	case OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
		n.Left.exprfmt(s, nprec, mode)
		fmt.Fprint(s, "[")
		low, high, max := n.SliceBounds()
		if low != nil {
			fmt.Fprint(s, low.modeString(mode))
		}
		fmt.Fprint(s, ":")
		if high != nil {
			fmt.Fprint(s, high.modeString(mode))
		}
		if n.Op.IsSlice3() {
			fmt.Fprint(s, ":")
			if max != nil {
				fmt.Fprint(s, max.modeString(mode))
			}
		}
		fmt.Fprint(s, "]")

	case OCOPY, OCOMPLEX:
		mode.Fprintf(s, "%#v(%v, %v)", n.Op, n.Left, n.Right)

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		ORUNESTR:
		if n.Type == nil || n.Type.Sym == nil {
			mode.Fprintf(s, "(%v)", n.Type)
		} else {
			mode.Fprintf(s, "%v", n.Type)
		}
		if n.Left != nil {
			mode.Fprintf(s, "(%v)", n.Left)
		} else {
			mode.Fprintf(s, "(%.v)", n.List)
		}

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
			mode.Fprintf(s, "%#v(%v)", n.Op, n.Left)
			return
		}
		if n.Isddd() {
			mode.Fprintf(s, "%#v(%.v...)", n.Op, n.List)
			return
		}
		mode.Fprintf(s, "%#v(%.v)", n.Op, n.List)

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		n.Left.exprfmt(s, nprec, mode)
		if n.Isddd() {
			mode.Fprintf(s, "(%.v...)", n.List)
			return
		}
		mode.Fprintf(s, "(%.v)", n.List)

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		if n.List.Len() != 0 { // pre-typecheck
			mode.Fprintf(s, "make(%v, %.v)", n.Type, n.List)
			return
		}
		if n.Right != nil {
			mode.Fprintf(s, "make(%v, %v, %v)", n.Type, n.Left, n.Right)
			return
		}
		if n.Left != nil && (n.Op == OMAKESLICE || !n.Left.Type.IsUntyped()) {
			mode.Fprintf(s, "make(%v, %v)", n.Type, n.Left)
			return
		}
		mode.Fprintf(s, "make(%v)", n.Type)

		// Unary
	case OPLUS,
		OMINUS,
		OADDR,
		OCOM,
		OIND,
		ONOT,
		ORECV:
		mode.Fprintf(s, "%#v", n.Op)
		if n.Left.Op == n.Op {
			fmt.Fprint(s, " ")
		}
		n.Left.exprfmt(s, nprec+1, mode)

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
		n.Left.exprfmt(s, nprec, mode)
		mode.Fprintf(s, " %#v ", n.Op)
		n.Right.exprfmt(s, nprec+1, mode)

	case OADDSTR:
		i := 0
		for _, n1 := range n.List.Slice() {
			if i != 0 {
				fmt.Fprint(s, " + ")
			}
			n1.exprfmt(s, nprec, mode)
			i++
		}

	case OCMPSTR, OCMPIFACE:
		n.Left.exprfmt(s, nprec, mode)
		// TODO(marvin): Fix Node.EType type union.
		mode.Fprintf(s, " %#v ", Op(n.Etype))
		n.Right.exprfmt(s, nprec+1, mode)

	default:
		mode.Fprintf(s, "<node %v>", n.Op)
	}
}

func (n *Node) nodefmt(s fmt.State, flag FmtFlag, mode fmtMode) {
	t := n.Type

	// We almost always want the original, except in export mode for literals.
	// This saves the importer some work, and avoids us having to redo some
	// special casing for package unsafe.
	if n.Op != OLITERAL && n.Orig != nil {
		n = n.Orig
	}

	if flag&FmtLong != 0 && t != nil {
		if t.Etype == TNIL {
			fmt.Fprint(s, "nil")
		} else {
			mode.Fprintf(s, "%v (type %v)", n, t)
		}
		return
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if opprec[n.Op] < 0 {
		n.stmtfmt(s, mode)
		return
	}

	n.exprfmt(s, 0, mode)
}

func (n *Node) nodedump(s fmt.State, flag FmtFlag, mode fmtMode) {
	recur := flag&FmtShort == 0

	if recur {
		indent(s)
		if dumpdepth > 40 {
			fmt.Fprint(s, "...")
			return
		}

		if n.Ninit.Len() != 0 {
			mode.Fprintf(s, "%v-init%v", n.Op, n.Ninit)
			indent(s)
		}
	}

	switch n.Op {
	default:
		mode.Fprintf(s, "%v%j", n.Op, n)

	case OINDREGSP:
		mode.Fprintf(s, "%v-SP%j", n.Op, n)

	case OLITERAL:
		mode.Fprintf(s, "%v-%v%j", n.Op, n.Val(), n)

	case ONAME, ONONAME:
		if n.Sym != nil {
			mode.Fprintf(s, "%v-%v%j", n.Op, n.Sym, n)
		} else {
			mode.Fprintf(s, "%v%j", n.Op, n)
		}
		if recur && n.Type == nil && n.Name != nil && n.Name.Param != nil && n.Name.Param.Ntype != nil {
			indent(s)
			mode.Fprintf(s, "%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}

	case OASOP:
		mode.Fprintf(s, "%v-%v%j", n.Op, Op(n.Etype), n)

	case OTYPE:
		mode.Fprintf(s, "%v %v%j type=%v", n.Op, n.Sym, n, n.Type)
		if recur && n.Type == nil && n.Name.Param.Ntype != nil {
			indent(s)
			mode.Fprintf(s, "%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}
	}

	if n.Sym != nil && n.Op != ONAME {
		mode.Fprintf(s, " %v", n.Sym)
	}

	if n.Type != nil {
		mode.Fprintf(s, " %v", n.Type)
	}

	if recur {
		if n.Left != nil {
			mode.Fprintf(s, "%v", n.Left)
		}
		if n.Right != nil {
			mode.Fprintf(s, "%v", n.Right)
		}
		if n.List.Len() != 0 {
			indent(s)
			mode.Fprintf(s, "%v-list%v", n.Op, n.List)
		}

		if n.Rlist.Len() != 0 {
			indent(s)
			mode.Fprintf(s, "%v-rlist%v", n.Op, n.Rlist)
		}

		if n.Nbody.Len() != 0 {
			indent(s)
			mode.Fprintf(s, "%v-body%v", n.Op, n.Nbody)
		}
	}
}

// "%S" suppresses qualifying with package
func (s *Sym) format(f fmt.State, verb rune, mode fmtMode) {
	switch verb {
	case 'v', 'S':
		fmt.Fprint(f, s.sconv(fmtFlag(f, verb), mode))

	default:
		fmt.Fprintf(f, "%%!%c(*Sym=%p)", verb, s)
	}
}

func (s *Sym) String() string                 { return s.sconv(0, FErr) }
func (s *Sym) modeString(mode fmtMode) string { return s.sconv(0, mode) }

// See #16897 before changing the implementation of sconv.
func (s *Sym) sconv(flag FmtFlag, mode fmtMode) string {
	if flag&FmtLong != 0 {
		panic("linksymfmt")
	}

	if s == nil {
		return "<S>"
	}

	if s.Name == "_" {
		return "_"
	}

	flag, mode = flag.update(mode)
	return s.symfmt(flag, mode)
}

func (t *Type) String() string {
	// This is an external entry point, so we pass depth 0 to tconv.
	// The implementation of tconv (including typefmt and fldconv)
	// must take care not to use a type in a formatting string
	// to avoid resetting the recursion counter.
	return t.tconv(0, FErr, 0)
}

func (t *Type) modeString(mode fmtMode, depth int) string {
	return t.tconv(0, mode, depth)
}

// ShortString generates a short description of t.
// It is used in autogenerated method names, reflection,
// and itab names.
func (t *Type) ShortString() string {
	return t.tconv(FmtLeft, FErr, 0)
}

// LongString generates a complete description of t.
// It is useful for reflection,
// or when a unique fingerprint or hash of a type is required.
func (t *Type) LongString() string {
	return t.tconv(FmtLeft|FmtUnsigned, FErr, 0)
}

func fldconv(f *Field, flag FmtFlag, mode fmtMode, depth int) string {
	if f == nil {
		return "<T>"
	}

	flag, mode = flag.update(mode)
	if mode == FTypeIdName {
		flag |= FmtUnsigned
	}

	var name string
	if flag&FmtShort == 0 {
		s := f.Sym

		// Take the name from the original, lest we substituted it with ~r%d or ~b%d.
		// ~r%d is a (formerly) unnamed result.
		if mode == FErr && f.Nname != nil {
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
				name = f.Nname.modeString(mode)
			} else if flag&FmtLong != 0 {
				name = mode.Sprintf("%0S", s)
				if !exportname(name) && flag&FmtUnsigned == 0 {
					name = s.modeString(mode) // qualify non-exported names (used on structs, not on funarg)
				}
			} else {
				name = s.modeString(mode)
			}
		}
	}

	var typ string
	if f.Isddd() {
		typ = "..." + f.Type.Elem().modeString(mode, depth)
	} else {
		typ = f.Type.modeString(mode, depth)
	}

	str := typ
	if name != "" {
		str = name + " " + typ
	}

	if flag&FmtShort == 0 && f.Funarg == FunargNone && f.Note != "" {
		str += " " + strconv.Quote(f.Note)
	}

	return str
}

// "%L"  print definition, not name
// "%S"  omit 'func' and receiver from function types, short type names
func (t *Type) format(s fmt.State, verb rune, mode fmtMode) {
	switch verb {
	case 'v', 'S', 'L':
		// This is an external entry point, so we pass depth 0 to tconv.
		// See comments in Type.String.
		fmt.Fprint(s, t.tconv(fmtFlag(s, verb), mode, 0))

	default:
		fmt.Fprintf(s, "%%!%c(*Type=%p)", verb, t)
	}
}

// See #16897 before changing the implementation of tconv.
func (t *Type) tconv(flag FmtFlag, mode fmtMode, depth int) string {
	if t == nil {
		return "<T>"
	}

	if depth > 100 {
		return "<...>"
	}

	flag, mode = flag.update(mode)
	if mode == FTypeIdName {
		flag |= FmtUnsigned
	}

	str := t.typefmt(flag, mode, depth+1)

	return str
}

func (n *Node) String() string                 { return fmt.Sprint(n) }
func (n *Node) modeString(mode fmtMode) string { return mode.Sprint(n) }

// "%L"  suffix with "(type %T)" where possible
// "%+S" in debug mode, don't recurse, no multiline output
func (n *Node) nconv(s fmt.State, flag FmtFlag, mode fmtMode) {
	if n == nil {
		fmt.Fprint(s, "<N>")
		return
	}

	flag, mode = flag.update(mode)

	switch mode {
	case FErr:
		n.nodefmt(s, flag, mode)

	case FDbg:
		dumpdepth++
		n.nodedump(s, flag, mode)
		dumpdepth--

	default:
		Fatalf("unhandled %%N mode: %d", mode)
	}
}

func (l Nodes) format(s fmt.State, verb rune, mode fmtMode) {
	switch verb {
	case 'v':
		l.hconv(s, fmtFlag(s, verb), mode)

	default:
		fmt.Fprintf(s, "%%!%c(Nodes)", verb)
	}
}

func (n Nodes) String() string {
	return fmt.Sprint(n)
}

// Flags: all those of %N plus '.': separate with comma's instead of semicolons.
func (l Nodes) hconv(s fmt.State, flag FmtFlag, mode fmtMode) {
	if l.Len() == 0 && mode == FDbg {
		fmt.Fprint(s, "<nil>")
		return
	}

	flag, mode = flag.update(mode)
	sep := "; "
	if mode == FDbg {
		sep = "\n"
	} else if flag&FmtComma != 0 {
		sep = ", "
	}

	for i, n := range l.Slice() {
		fmt.Fprint(s, n.modeString(mode))
		if i+1 < l.Len() {
			fmt.Fprint(s, sep)
		}
	}
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
