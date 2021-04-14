// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
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
//	%v *types.Sym		Symbols
//	%S              unqualified identifier in any mode
//		Flags:  +,- #: mode (see below)
//			0: in export mode: unqualified identifier if exported, qualified if not
//
//	%v *types.Type	Types
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

// *types.Sym, *types.Type, and *Node types use the flags below to set the format mode
const (
	FErr fmtMode = iota
	FDbg
	FTypeId
	FTypeIdName // same as FTypeId, but use package name instead of prefix
)

// The mode flags '+', '-', and '#' are sticky; they persist through
// recursions of *Node, *types.Type, and *types.Sym values. The ' ' flag is
// sticky only on *types.Type recursions and only used in %-/*types.Sym mode.
//
// Example: given a *types.Sym: %+v %#v %-v print an identifier properly qualified for debug/export/internal mode

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
// *types.Type:
//   %#v    Go format
//   %#L    type definition instead of name
//   %#S    omit "func" and receiver in function signature
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
	OBITNOT:   "^",
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
	OINLMARK:  "inlmark",
	ODEREF:    "*",
	OLEN:      "len",
	OLE:       "<=",
	OLSH:      "<<",
	OLT:       "<",
	OMAKE:     "make",
	ONEG:      "-",
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
		if int(o) < len(goopnames) && goopnames[o] != "" {
			fmt.Fprint(s, goopnames[o])
			return
		}
	}

	// 'o.String()' instead of just 'o' to avoid infinite recursion
	fmt.Fprint(s, o.String())
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

	fmtTypeErr        types.Type
	fmtTypeDbg        types.Type
	fmtTypeTypeId     types.Type
	fmtTypeTypeIdName types.Type

	fmtSymErr        types.Sym
	fmtSymDbg        types.Sym
	fmtSymTypeId     types.Sym
	fmtSymTypeIdName types.Sym

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

func (t *fmtTypeErr) Format(s fmt.State, verb rune) { typeFormat((*types.Type)(t), s, verb, FErr) }
func (t *fmtTypeDbg) Format(s fmt.State, verb rune) { typeFormat((*types.Type)(t), s, verb, FDbg) }
func (t *fmtTypeTypeId) Format(s fmt.State, verb rune) {
	typeFormat((*types.Type)(t), s, verb, FTypeId)
}
func (t *fmtTypeTypeIdName) Format(s fmt.State, verb rune) {
	typeFormat((*types.Type)(t), s, verb, FTypeIdName)
}

// func (t *types.Type) Format(s fmt.State, verb rune)     // in package types

func (y *fmtSymErr) Format(s fmt.State, verb rune)    { symFormat((*types.Sym)(y), s, verb, FErr) }
func (y *fmtSymDbg) Format(s fmt.State, verb rune)    { symFormat((*types.Sym)(y), s, verb, FDbg) }
func (y *fmtSymTypeId) Format(s fmt.State, verb rune) { symFormat((*types.Sym)(y), s, verb, FTypeId) }
func (y *fmtSymTypeIdName) Format(s fmt.State, verb rune) {
	symFormat((*types.Sym)(y), s, verb, FTypeIdName)
}

// func (y *types.Sym) Format(s fmt.State, verb rune)            // in package types  { y.format(s, verb, FErr) }

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
			case *types.Type:
				args[i] = (*fmtTypeErr)(arg)
			case *types.Sym:
				args[i] = (*fmtSymErr)(arg)
			case Nodes:
				args[i] = fmtNodesErr(arg)
			case Val, int32, int64, string, types.EType:
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
			case *types.Type:
				args[i] = (*fmtTypeDbg)(arg)
			case *types.Sym:
				args[i] = (*fmtSymDbg)(arg)
			case Nodes:
				args[i] = fmtNodesDbg(arg)
			case Val, int32, int64, string, types.EType:
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
			case *types.Type:
				args[i] = (*fmtTypeTypeId)(arg)
			case *types.Sym:
				args[i] = (*fmtSymTypeId)(arg)
			case Nodes:
				args[i] = fmtNodesTypeId(arg)
			case Val, int32, int64, string, types.EType:
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
			case *types.Type:
				args[i] = (*fmtTypeTypeIdName)(arg)
			case *types.Sym:
				args[i] = (*fmtSymTypeIdName)(arg)
			case Nodes:
				args[i] = fmtNodesTypeIdName(arg)
			case Val, int32, int64, string, types.EType:
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

	// Useful to see which nodes in a Node Dump/dumplist are actually identical
	if Debug_dumpptrs != 0 {
		fmt.Fprintf(s, " p(%p)", n)
	}
	if c == 0 && n.Name != nil && n.Name.Vargen != 0 {
		fmt.Fprintf(s, " g(%d)", n.Name.Vargen)
	}

	if Debug_dumpptrs != 0 && c == 0 && n.Name != nil && n.Name.Defn != nil {
		// Useful to see where Defn is set and what node it points to
		fmt.Fprintf(s, " defn(%p)", n.Name.Defn)
	}

	if n.Pos.IsKnown() {
		pfx := ""
		switch n.Pos.IsStmt() {
		case src.PosNotStmt:
			pfx = "_" // "-" would be confusing
		case src.PosIsStmt:
			pfx = "+"
		}
		fmt.Fprintf(s, " l(%s%d)", pfx, n.Pos.Line())
	}

	if c == 0 && n.Xoffset != BADWIDTH {
		fmt.Fprintf(s, " x(%d)", n.Xoffset)
	}

	if n.Class() != 0 {
		fmt.Fprintf(s, " class(%v)", n.Class())
	}

	if n.Colas() {
		fmt.Fprintf(s, " colas(%v)", n.Colas())
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

	if e, ok := n.Opt().(*EscLocation); ok && e.loopDepth != 0 {
		fmt.Fprintf(s, " ld(%d)", e.loopDepth)
	}

	if c == 0 && n.Typecheck() != 0 {
		fmt.Fprintf(s, " tc(%d)", n.Typecheck())
	}

	if n.IsDDD() {
		fmt.Fprintf(s, " isddd(%v)", n.IsDDD())
	}

	if n.Implicit() {
		fmt.Fprintf(s, " implicit(%v)", n.Implicit())
	}

	if n.Embedded() {
		fmt.Fprintf(s, " embedded")
	}

	if n.Op == ONAME {
		if n.Name.Addrtaken() {
			fmt.Fprint(s, " addrtaken")
		}
		if n.Name.Assigned() {
			fmt.Fprint(s, " assigned")
		}
		if n.Name.IsClosureVar() {
			fmt.Fprint(s, " closurevar")
		}
		if n.Name.Captured() {
			fmt.Fprint(s, " captured")
		}
		if n.Name.IsOutputParamHeapAddr() {
			fmt.Fprint(s, " outputparamheapaddr")
		}
	}
	if n.Bounded() {
		fmt.Fprint(s, " bounded")
	}
	if n.NonNil() {
		fmt.Fprint(s, " nonnil")
	}

	if c == 0 && n.HasCall() {
		fmt.Fprint(s, " hascall")
	}

	if c == 0 && n.Name != nil && n.Name.Used() {
		fmt.Fprint(s, " used")
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
				fmt.Fprint(s, u.String())
				return
			}
			fmt.Fprint(s, u.GoString())
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
			fmt.Fprint(s, u.String())
			return
		}
		fmt.Fprint(s, u.GoString())
		return

	case *Mpcplx:
		if flag&FmtSharp != 0 {
			fmt.Fprint(s, u.String())
			return
		}
		fmt.Fprint(s, u.GoString())
		return

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

func symfmt(b *bytes.Buffer, s *types.Sym, flag FmtFlag, mode fmtMode) {
	if flag&FmtShort == 0 {
		switch mode {
		case FErr: // This is for the user
			if s.Pkg == builtinpkg || s.Pkg == localpkg {
				b.WriteString(s.Name)
				return
			}

			// If the name was used by multiple packages, display the full path,
			if s.Pkg.Name != "" && numImport[s.Pkg.Name] > 1 {
				fmt.Fprintf(b, "%q.%s", s.Pkg.Path, s.Name)
				return
			}
			b.WriteString(s.Pkg.Name)
			b.WriteByte('.')
			b.WriteString(s.Name)
			return

		case FDbg:
			b.WriteString(s.Pkg.Name)
			b.WriteByte('.')
			b.WriteString(s.Name)
			return

		case FTypeIdName:
			// dcommontype, typehash
			b.WriteString(s.Pkg.Name)
			b.WriteByte('.')
			b.WriteString(s.Name)
			return

		case FTypeId:
			// (methodsym), typesym, weaksym
			b.WriteString(s.Pkg.Prefix)
			b.WriteByte('.')
			b.WriteString(s.Name)
			return
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
			fmt.Fprintf(b, "@%q.%s", s.Pkg.Path, name)
			return
		}

		b.WriteString(name)
		return
	}

	b.WriteString(s.Name)
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

var fmtBufferPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

func tconv(t *types.Type, flag FmtFlag, mode fmtMode) string {
	buf := fmtBufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer fmtBufferPool.Put(buf)

	tconv2(buf, t, flag, mode, nil)
	return types.InternString(buf.Bytes())
}

// tconv2 writes a string representation of t to b.
// flag and mode control exactly what is printed.
// Any types x that are already in the visited map get printed as @%d where %d=visited[x].
// See #16897 before changing the implementation of tconv.
func tconv2(b *bytes.Buffer, t *types.Type, flag FmtFlag, mode fmtMode, visited map[*types.Type]int) {
	if off, ok := visited[t]; ok {
		// We've seen this type before, so we're trying to print it recursively.
		// Print a reference to it instead.
		fmt.Fprintf(b, "@%d", off)
		return
	}
	if t == nil {
		b.WriteString("<T>")
		return
	}
	if t.Etype == types.TSSA {
		b.WriteString(t.Extra.(string))
		return
	}
	if t.Etype == types.TTUPLE {
		b.WriteString(t.FieldType(0).String())
		b.WriteByte(',')
		b.WriteString(t.FieldType(1).String())
		return
	}

	if t.Etype == types.TRESULTS {
		tys := t.Extra.(*types.Results).Types
		for i, et := range tys {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(et.String())
		}
		return
	}

	flag, mode = flag.update(mode)
	if mode == FTypeIdName {
		flag |= FmtUnsigned
	}
	if t == types.Bytetype || t == types.Runetype {
		// in %-T mode collapse rune and byte with their originals.
		switch mode {
		case FTypeIdName, FTypeId:
			t = types.Types[t.Etype]
		default:
			sconv2(b, t.Sym, FmtShort, mode)
			return
		}
	}
	if t == types.Errortype {
		b.WriteString("error")
		return
	}

	// Unless the 'L' flag was specified, if the type has a name, just print that name.
	if flag&FmtLong == 0 && t.Sym != nil && t != types.Types[t.Etype] {
		switch mode {
		case FTypeId, FTypeIdName:
			if flag&FmtShort != 0 {
				if t.Vargen != 0 {
					sconv2(b, t.Sym, FmtShort, mode)
					fmt.Fprintf(b, "·%d", t.Vargen)
					return
				}
				sconv2(b, t.Sym, FmtShort, mode)
				return
			}

			if mode == FTypeIdName {
				sconv2(b, t.Sym, FmtUnsigned, mode)
				return
			}

			if t.Sym.Pkg == localpkg && t.Vargen != 0 {
				b.WriteString(mode.Sprintf("%v·%d", t.Sym, t.Vargen))
				return
			}
		}

		sconv2(b, t.Sym, 0, mode)
		return
	}

	if int(t.Etype) < len(basicnames) && basicnames[t.Etype] != "" {
		var name string
		switch t {
		case types.UntypedBool:
			name = "untyped bool"
		case types.UntypedString:
			name = "untyped string"
		case types.UntypedInt:
			name = "untyped int"
		case types.UntypedRune:
			name = "untyped rune"
		case types.UntypedFloat:
			name = "untyped float"
		case types.UntypedComplex:
			name = "untyped complex"
		default:
			name = basicnames[t.Etype]
		}
		b.WriteString(name)
		return
	}

	if mode == FDbg {
		b.WriteString(t.Etype.String())
		b.WriteByte('-')
		tconv2(b, t, flag, FErr, visited)
		return
	}

	// At this point, we might call tconv2 recursively. Add the current type to the visited list so we don't
	// try to print it recursively.
	// We record the offset in the result buffer where the type's text starts. This offset serves as a reference
	// point for any later references to the same type.
	// Note that we remove the type from the visited map as soon as the recursive call is done.
	// This prevents encoding types like map[*int]*int as map[*int]@4. (That encoding would work,
	// but I'd like to use the @ notation only when strictly necessary.)
	if visited == nil {
		visited = map[*types.Type]int{}
	}
	visited[t] = b.Len()
	defer delete(visited, t)

	switch t.Etype {
	case TPTR:
		b.WriteByte('*')
		switch mode {
		case FTypeId, FTypeIdName:
			if flag&FmtShort != 0 {
				tconv2(b, t.Elem(), FmtShort, mode, visited)
				return
			}
		}
		tconv2(b, t.Elem(), 0, mode, visited)

	case TARRAY:
		b.WriteByte('[')
		b.WriteString(strconv.FormatInt(t.NumElem(), 10))
		b.WriteByte(']')
		tconv2(b, t.Elem(), 0, mode, visited)

	case TSLICE:
		b.WriteString("[]")
		tconv2(b, t.Elem(), 0, mode, visited)

	case TCHAN:
		switch t.ChanDir() {
		case types.Crecv:
			b.WriteString("<-chan ")
			tconv2(b, t.Elem(), 0, mode, visited)
		case types.Csend:
			b.WriteString("chan<- ")
			tconv2(b, t.Elem(), 0, mode, visited)
		default:
			b.WriteString("chan ")
			if t.Elem() != nil && t.Elem().IsChan() && t.Elem().Sym == nil && t.Elem().ChanDir() == types.Crecv {
				b.WriteByte('(')
				tconv2(b, t.Elem(), 0, mode, visited)
				b.WriteByte(')')
			} else {
				tconv2(b, t.Elem(), 0, mode, visited)
			}
		}

	case TMAP:
		b.WriteString("map[")
		tconv2(b, t.Key(), 0, mode, visited)
		b.WriteByte(']')
		tconv2(b, t.Elem(), 0, mode, visited)

	case TINTER:
		if t.IsEmptyInterface() {
			b.WriteString("interface {}")
			break
		}
		b.WriteString("interface {")
		for i, f := range t.Fields().Slice() {
			if i != 0 {
				b.WriteByte(';')
			}
			b.WriteByte(' ')
			switch {
			case f.Sym == nil:
				// Check first that a symbol is defined for this type.
				// Wrong interface definitions may have types lacking a symbol.
				break
			case types.IsExported(f.Sym.Name):
				sconv2(b, f.Sym, FmtShort, mode)
			default:
				flag1 := FmtLeft
				if flag&FmtUnsigned != 0 {
					flag1 = FmtUnsigned
				}
				sconv2(b, f.Sym, flag1, mode)
			}
			tconv2(b, f.Type, FmtShort, mode, visited)
		}
		if t.NumFields() != 0 {
			b.WriteByte(' ')
		}
		b.WriteByte('}')

	case TFUNC:
		if flag&FmtShort != 0 {
			// no leading func
		} else {
			if t.Recv() != nil {
				b.WriteString("method")
				tconv2(b, t.Recvs(), 0, mode, visited)
				b.WriteByte(' ')
			}
			b.WriteString("func")
		}
		tconv2(b, t.Params(), 0, mode, visited)

		switch t.NumResults() {
		case 0:
			// nothing to do

		case 1:
			b.WriteByte(' ')
			tconv2(b, t.Results().Field(0).Type, 0, mode, visited) // struct->field->field's type

		default:
			b.WriteByte(' ')
			tconv2(b, t.Results(), 0, mode, visited)
		}

	case TSTRUCT:
		if m := t.StructType().Map; m != nil {
			mt := m.MapType()
			// Format the bucket struct for map[x]y as map.bucket[x]y.
			// This avoids a recursive print that generates very long names.
			switch t {
			case mt.Bucket:
				b.WriteString("map.bucket[")
			case mt.Hmap:
				b.WriteString("map.hdr[")
			case mt.Hiter:
				b.WriteString("map.iter[")
			default:
				Fatalf("unknown internal map type")
			}
			tconv2(b, m.Key(), 0, mode, visited)
			b.WriteByte(']')
			tconv2(b, m.Elem(), 0, mode, visited)
			break
		}

		if funarg := t.StructType().Funarg; funarg != types.FunargNone {
			b.WriteByte('(')
			var flag1 FmtFlag
			switch mode {
			case FTypeId, FTypeIdName, FErr:
				// no argument names on function signature, and no "noescape"/"nosplit" tags
				flag1 = FmtShort
			}
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					b.WriteString(", ")
				}
				fldconv(b, f, flag1, mode, visited, funarg)
			}
			b.WriteByte(')')
		} else {
			b.WriteString("struct {")
			for i, f := range t.Fields().Slice() {
				if i != 0 {
					b.WriteByte(';')
				}
				b.WriteByte(' ')
				fldconv(b, f, FmtLong, mode, visited, funarg)
			}
			if t.NumFields() != 0 {
				b.WriteByte(' ')
			}
			b.WriteByte('}')
		}

	case TFORW:
		b.WriteString("undefined")
		if t.Sym != nil {
			b.WriteByte(' ')
			sconv2(b, t.Sym, 0, mode)
		}

	case TUNSAFEPTR:
		b.WriteString("unsafe.Pointer")

	case Txxx:
		b.WriteString("Txxx")
	default:
		// Don't know how to handle - fall back to detailed prints.
		b.WriteString(mode.Sprintf("%v <%v>", t.Etype, t.Sym))
	}
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
		if n.Sym != nil {
			mode.Fprintf(s, "%v %v", n.Sym, n.Left)
		} else {
			mode.Fprintf(s, "%v", n.Left)
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
			if n.SubOp() == OADD {
				mode.Fprintf(s, "%v++", n.Left)
			} else {
				mode.Fprintf(s, "%v--", n.Left)
			}
			break
		}

		mode.Fprintf(s, "%v %#v= %v", n.Left, n.SubOp(), n.Right)

	case OAS2:
		if n.Colas() && !complexinit {
			mode.Fprintf(s, "%.v := %.v", n.List, n.Rlist)
			break
		}
		fallthrough

	case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		mode.Fprintf(s, "%.v = %v", n.List, n.Right)

	case ORETURN:
		mode.Fprintf(s, "return %.v", n.List)

	case ORETJMP:
		mode.Fprintf(s, "retjmp %v", n.Sym)

	case OINLMARK:
		mode.Fprintf(s, "inlmark %d", n.Xoffset)

	case OGO:
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

		if n.Op == OFORUNTIL && n.List.Len() != 0 {
			mode.Fprintf(s, "; %v", n.List)
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

	case OCASE:
		if n.List.Len() != 0 {
			mode.Fprintf(s, "case %.v", n.List)
		} else {
			fmt.Fprint(s, "default")
		}
		mode.Fprintf(s, ": %v", n.Nbody)

	case OBREAK, OCONTINUE, OGOTO, OFALL:
		if n.Sym != nil {
			mode.Fprintf(s, "%#v %v", n.Op, n.Sym)
		} else {
			mode.Fprintf(s, "%#v", n.Op)
		}

	case OEMPTY:
		break

	case OLABEL:
		mode.Fprintf(s, "%v: ", n.Sym)
	}

	if extrablock {
		fmt.Fprint(s, "}")
	}
}

var opprec = []int{
	OALIGNOF:       8,
	OAPPEND:        8,
	OBYTES2STR:     8,
	OARRAYLIT:      8,
	OSLICELIT:      8,
	ORUNES2STR:     8,
	OCALLFUNC:      8,
	OCALLINTER:     8,
	OCALLMETH:      8,
	OCALL:          8,
	OCAP:           8,
	OCLOSE:         8,
	OCONVIFACE:     8,
	OCONVNOP:       8,
	OCONV:          8,
	OCOPY:          8,
	ODELETE:        8,
	OGETG:          8,
	OLEN:           8,
	OLITERAL:       8,
	OMAKESLICE:     8,
	OMAKESLICECOPY: 8,
	OMAKE:          8,
	OMAPLIT:        8,
	ONAME:          8,
	ONEW:           8,
	ONONAME:        8,
	OOFFSETOF:      8,
	OPACK:          8,
	OPANIC:         8,
	OPAREN:         8,
	OPRINTN:        8,
	OPRINT:         8,
	ORUNESTR:       8,
	OSIZEOF:        8,
	OSTR2BYTES:     8,
	OSTR2RUNES:     8,
	OSTRUCTLIT:     8,
	OTARRAY:        8,
	OTCHAN:         8,
	OTFUNC:         8,
	OTINTER:        8,
	OTMAP:          8,
	OTSTRUCT:       8,
	OINDEXMAP:      8,
	OINDEX:         8,
	OSLICE:         8,
	OSLICESTR:      8,
	OSLICEARR:      8,
	OSLICE3:        8,
	OSLICE3ARR:     8,
	OSLICEHEADER:   8,
	ODOTINTER:      8,
	ODOTMETH:       8,
	ODOTPTR:        8,
	ODOTTYPE2:      8,
	ODOTTYPE:       8,
	ODOT:           8,
	OXDOT:          8,
	OCALLPART:      8,
	OPLUS:          7,
	ONOT:           7,
	OBITNOT:        7,
	ONEG:           7,
	OADDR:          7,
	ODEREF:         7,
	ORECV:          7,
	OMUL:           6,
	ODIV:           6,
	OMOD:           6,
	OLSH:           6,
	ORSH:           6,
	OAND:           6,
	OANDNOT:        6,
	OADD:           5,
	OSUB:           5,
	OOR:            5,
	OXOR:           5,
	OEQ:            4,
	OLT:            4,
	OLE:            4,
	OGE:            4,
	OGT:            4,
	ONE:            4,
	OSEND:          3,
	OANDAND:        2,
	OOROR:          1,

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
	OGO:         -1,
	ORANGE:      -1,
	ORETURN:     -1,
	OSELECT:     -1,
	OSWITCH:     -1,

	OEND: 0,
}

func (n *Node) exprfmt(s fmt.State, prec int, mode fmtMode) {
	for n != nil && n.Implicit() && (n.Op == ODEREF || n.Op == OADDR) {
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

	case OLITERAL: // this is a bit of a mess
		if mode == FErr {
			if n.Orig != nil && n.Orig != n {
				n.Orig.exprfmt(s, prec, mode)
				return
			}
			if n.Sym != nil {
				fmt.Fprint(s, smodeString(n.Sym, mode))
				return
			}
		}
		if n.Val().Ctype() == CTNIL && n.Orig != nil && n.Orig != n {
			n.Orig.exprfmt(s, prec, mode)
			return
		}
		if n.Type != nil && !n.Type.IsUntyped() {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if n.Type.IsPtr() || (n.Type.IsChan() && n.Type.ChanDir() == types.Crecv) {
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
		fmt.Fprint(s, smodeString(n.Sym, mode))

	case OTYPE:
		if n.Type == nil && n.Sym != nil {
			fmt.Fprint(s, smodeString(n.Sym, mode))
			return
		}
		mode.Fprintf(s, "%v", n.Type)

	case OTARRAY:
		if n.Left != nil {
			mode.Fprintf(s, "[%v]%v", n.Left, n.Right)
			return
		}
		mode.Fprintf(s, "[]%v", n.Right) // happens before typecheck

	case OTMAP:
		mode.Fprintf(s, "map[%v]%v", n.Left, n.Right)

	case OTCHAN:
		switch n.TChanDir() {
		case types.Crecv:
			mode.Fprintf(s, "<-chan %v", n.Left)

		case types.Csend:
			mode.Fprintf(s, "chan<- %v", n.Left)

		default:
			if n.Left != nil && n.Left.Op == OTCHAN && n.Left.Sym == nil && n.Left.TChanDir() == types.Crecv {
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
		if mode == FErr {
			if n.Implicit() {
				mode.Fprintf(s, "... argument")
				return
			}
			if n.Right != nil {
				mode.Fprintf(s, "%v{%s}", n.Right, ellipsisIf(n.List.Len() != 0))
				return
			}

			fmt.Fprint(s, "composite literal")
			return
		}
		mode.Fprintf(s, "(%v{ %.v })", n.Right, n.List)

	case OPTRLIT:
		mode.Fprintf(s, "&%v", n.Left)

	case OSTRUCTLIT, OARRAYLIT, OSLICELIT, OMAPLIT:
		if mode == FErr {
			mode.Fprintf(s, "%v{%s}", n.Type, ellipsisIf(n.List.Len() != 0))
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

	case OSLICEHEADER:
		if n.List.Len() != 2 {
			Fatalf("bad OSLICEHEADER list length %d", n.List.Len())
		}
		mode.Fprintf(s, "sliceheader{%v,%v,%v}", n.Left, n.List.First(), n.List.Second())

	case OCOMPLEX, OCOPY:
		if n.Left != nil {
			mode.Fprintf(s, "%#v(%v, %v)", n.Op, n.Left, n.Right)
		} else {
			mode.Fprintf(s, "%#v(%.v)", n.Op, n.List)
		}

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OBYTES2STR,
		ORUNES2STR,
		OSTR2BYTES,
		OSTR2RUNES,
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
		if n.IsDDD() {
			mode.Fprintf(s, "%#v(%.v...)", n.Op, n.List)
			return
		}
		mode.Fprintf(s, "%#v(%.v)", n.Op, n.List)

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		n.Left.exprfmt(s, nprec, mode)
		if n.IsDDD() {
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

	case OMAKESLICECOPY:
		mode.Fprintf(s, "makeslicecopy(%v, %v, %v)", n.Type, n.Left, n.Right)

	case OPLUS, ONEG, OADDR, OBITNOT, ODEREF, ONOT, ORECV:
		// Unary
		mode.Fprintf(s, "%#v", n.Op)
		if n.Left != nil && n.Left.Op == n.Op {
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
		for i, n1 := range n.List.Slice() {
			if i != 0 {
				fmt.Fprint(s, " + ")
			}
			n1.exprfmt(s, nprec, mode)
		}
	case ODDD:
		mode.Fprintf(s, "...")
	default:
		mode.Fprintf(s, "<node %v>", n.Op)
	}
}

func (n *Node) nodefmt(s fmt.State, flag FmtFlag, mode fmtMode) {
	t := n.Type

	// We almost always want the original.
	// TODO(gri) Why the special case for OLITERAL?
	if n.Op != OLITERAL && n.Orig != nil {
		n = n.Orig
	}

	if flag&FmtLong != 0 && t != nil {
		if t.Etype == TNIL {
			fmt.Fprint(s, "nil")
		} else if n.Op == ONAME && n.Name.AutoTemp() {
			mode.Fprintf(s, "%v value", t)
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
		mode.Fprintf(s, "%v-%v%j", n.Op, n.SubOp(), n)

	case OTYPE:
		mode.Fprintf(s, "%v %v%j type=%v", n.Op, n.Sym, n, n.Type)
		if recur && n.Type == nil && n.Name != nil && n.Name.Param != nil && n.Name.Param.Ntype != nil {
			indent(s)
			mode.Fprintf(s, "%v-ntype%v", n.Op, n.Name.Param.Ntype)
		}
	}

	if n.Op == OCLOSURE && n.Func.Closure != nil && n.Func.Closure.Func.Nname.Sym != nil {
		mode.Fprintf(s, " fnName %v", n.Func.Closure.Func.Nname.Sym)
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
		if n.Func != nil && n.Func.Closure != nil && n.Func.Closure.Nbody.Len() != 0 {
			indent(s)
			// The function associated with a closure
			mode.Fprintf(s, "%v-clofunc%v", n.Op, n.Func.Closure)
		}
		if n.Func != nil && n.Func.Dcl != nil && len(n.Func.Dcl) != 0 {
			indent(s)
			// The dcls for a func or closure
			mode.Fprintf(s, "%v-dcl%v", n.Op, asNodes(n.Func.Dcl))
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
func symFormat(s *types.Sym, f fmt.State, verb rune, mode fmtMode) {
	switch verb {
	case 'v', 'S':
		fmt.Fprint(f, sconv(s, fmtFlag(f, verb), mode))

	default:
		fmt.Fprintf(f, "%%!%c(*types.Sym=%p)", verb, s)
	}
}

func smodeString(s *types.Sym, mode fmtMode) string { return sconv(s, 0, mode) }

// See #16897 before changing the implementation of sconv.
func sconv(s *types.Sym, flag FmtFlag, mode fmtMode) string {
	if flag&FmtLong != 0 {
		panic("linksymfmt")
	}

	if s == nil {
		return "<S>"
	}

	if s.Name == "_" {
		return "_"
	}
	buf := fmtBufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer fmtBufferPool.Put(buf)

	flag, mode = flag.update(mode)
	symfmt(buf, s, flag, mode)
	return types.InternString(buf.Bytes())
}

func sconv2(b *bytes.Buffer, s *types.Sym, flag FmtFlag, mode fmtMode) {
	if flag&FmtLong != 0 {
		panic("linksymfmt")
	}
	if s == nil {
		b.WriteString("<S>")
		return
	}
	if s.Name == "_" {
		b.WriteString("_")
		return
	}

	flag, mode = flag.update(mode)
	symfmt(b, s, flag, mode)
}

func fldconv(b *bytes.Buffer, f *types.Field, flag FmtFlag, mode fmtMode, visited map[*types.Type]int, funarg types.Funarg) {
	if f == nil {
		b.WriteString("<T>")
		return
	}
	flag, mode = flag.update(mode)
	if mode == FTypeIdName {
		flag |= FmtUnsigned
	}

	var name string
	if flag&FmtShort == 0 {
		s := f.Sym

		// Take the name from the original.
		if mode == FErr {
			s = origSym(s)
		}

		if s != nil && f.Embedded == 0 {
			if funarg != types.FunargNone {
				name = asNode(f.Nname).modeString(mode)
			} else if flag&FmtLong != 0 {
				name = mode.Sprintf("%0S", s)
				if !types.IsExported(name) && flag&FmtUnsigned == 0 {
					name = smodeString(s, mode) // qualify non-exported names (used on structs, not on funarg)
				}
			} else {
				name = smodeString(s, mode)
			}
		}
	}

	if name != "" {
		b.WriteString(name)
		b.WriteString(" ")
	}

	if f.IsDDD() {
		var et *types.Type
		if f.Type != nil {
			et = f.Type.Elem()
		}
		b.WriteString("...")
		tconv2(b, et, 0, mode, visited)
	} else {
		tconv2(b, f.Type, 0, mode, visited)
	}

	if flag&FmtShort == 0 && funarg == types.FunargNone && f.Note != "" {
		b.WriteString(" ")
		b.WriteString(strconv.Quote(f.Note))
	}
}

// "%L"  print definition, not name
// "%S"  omit 'func' and receiver from function types, short type names
func typeFormat(t *types.Type, s fmt.State, verb rune, mode fmtMode) {
	switch verb {
	case 'v', 'S', 'L':
		fmt.Fprint(s, tconv(t, fmtFlag(s, verb), mode))
	default:
		fmt.Fprintf(s, "%%!%c(*Type=%p)", verb, t)
	}
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

func fdumplist(w io.Writer, s string, l Nodes) {
	fmt.Fprintf(w, "%s%+v\n", s, l)
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

func ellipsisIf(b bool) string {
	if b {
		return "..."
	}
	return ""
}
