// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"bytes"
	"fmt"
	"go/constant"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

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

type FmtMode int

const (
	FErr FmtMode = iota
	FDbg
	FTypeId
	FTypeIdName // same as FTypeId, but use package name instead of prefix
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
		base.Fatalf("FmtUnsigned in format string")
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

// update returns the results of applying f to mode.
func (f FmtFlag) update(mode FmtMode) (FmtFlag, FmtMode) {
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

func (m FmtMode) Fprintf(s fmt.State, format string, args ...interface{}) {
	m.prepareArgs(args)
	fmt.Fprintf(s, format, args...)
}

func (m FmtMode) Sprintf(format string, args ...interface{}) string {
	m.prepareArgs(args)
	return fmt.Sprintf(format, args...)
}

func (m FmtMode) Sprint(args ...interface{}) string {
	m.prepareArgs(args)
	return fmt.Sprint(args...)
}

func (m FmtMode) prepareArgs(args []interface{}) {
	for i, arg := range args {
		switch arg := arg.(type) {
		case Op:
			args[i] = &fmtOp{arg, m}
		case Node:
			args[i] = &fmtNode{arg, m}
		case nil:
			args[i] = &fmtNode{nil, m} // assume this was a node interface
		case *types.Type:
			args[i] = &fmtType{arg, m}
		case *types.Sym:
			args[i] = &fmtSym{arg, m}
		case Nodes:
			args[i] = &fmtNodes{arg, m}
		case int32, int64, string, types.Kind, constant.Value:
			// OK: printing these types doesn't depend on mode
		default:
			base.Fatalf("mode.prepareArgs type %T", arg)
		}
	}
}

// Op

var OpNames = []string{
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
	if int(o) < len(OpNames) && OpNames[o] != "" {
		return OpNames[o]
	}
	return o.String()
}

type fmtOp struct {
	x Op
	m FmtMode
}

func (f *fmtOp) Format(s fmt.State, verb rune) { f.x.Format(s, verb) }

func (o Op) Format(s fmt.State, verb rune) {
	switch verb {
	default:
		fmt.Fprintf(s, "%%!%c(Op=%d)", verb, int(o))
	case 'v':
		if s.Flag('+') {
			// %+v is OMUL instead of "*"
			io.WriteString(s, o.String())
			return
		}
		io.WriteString(s, o.GoString())
	}
}

// Val

func FmtConst(v constant.Value, flag FmtFlag) string {
	if flag&FmtSharp == 0 && v.Kind() == constant.Complex {
		real, imag := constant.Real(v), constant.Imag(v)

		var re string
		sre := constant.Sign(real)
		if sre != 0 {
			re = real.String()
		}

		var im string
		sim := constant.Sign(imag)
		if sim != 0 {
			im = imag.String()
		}

		switch {
		case sre == 0 && sim == 0:
			return "0"
		case sre == 0:
			return im + "i"
		case sim == 0:
			return re
		case sim < 0:
			return fmt.Sprintf("(%s%si)", re, im)
		default:
			return fmt.Sprintf("(%s+%si)", re, im)
		}
	}

	return v.String()
}

// Sym

// numImport tracks how often a package with a given name is imported.
// It is used to provide a better error message (by using the package
// path to disambiguate) if a package that appears multiple times with
// the same name appears in an error message.
var NumImport = make(map[string]int)

type fmtSym struct {
	x *types.Sym
	m FmtMode
}

func (f *fmtSym) Format(s fmt.State, verb rune) { symFormat(f.x, s, verb, f.m) }

// "%S" suppresses qualifying with package
func symFormat(s *types.Sym, f fmt.State, verb rune, mode FmtMode) {
	switch verb {
	case 'v', 'S':
		if verb == 'v' && f.Flag('+') {
			mode = FDbg
		}
		fmt.Fprint(f, sconv(s, fmtFlag(f, verb), mode))

	default:
		fmt.Fprintf(f, "%%!%c(*types.Sym=%p)", verb, s)
	}
}

func smodeString(s *types.Sym, mode FmtMode) string { return sconv(s, 0, mode) }

// See #16897 before changing the implementation of sconv.
func sconv(s *types.Sym, flag FmtFlag, mode FmtMode) string {
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

func sconv2(b *bytes.Buffer, s *types.Sym, flag FmtFlag, mode FmtMode) {
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

func symfmt(b *bytes.Buffer, s *types.Sym, flag FmtFlag, mode FmtMode) {
	if flag&FmtShort == 0 {
		switch mode {
		case FErr: // This is for the user
			if s.Pkg == BuiltinPkg || s.Pkg == LocalPkg {
				b.WriteString(s.Name)
				return
			}

			// If the name was used by multiple packages, display the full path,
			if s.Pkg.Name != "" && NumImport[s.Pkg.Name] > 1 {
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

// Type

var BasicTypeNames = []string{
	types.TINT:        "int",
	types.TUINT:       "uint",
	types.TINT8:       "int8",
	types.TUINT8:      "uint8",
	types.TINT16:      "int16",
	types.TUINT16:     "uint16",
	types.TINT32:      "int32",
	types.TUINT32:     "uint32",
	types.TINT64:      "int64",
	types.TUINT64:     "uint64",
	types.TUINTPTR:    "uintptr",
	types.TFLOAT32:    "float32",
	types.TFLOAT64:    "float64",
	types.TCOMPLEX64:  "complex64",
	types.TCOMPLEX128: "complex128",
	types.TBOOL:       "bool",
	types.TANY:        "any",
	types.TSTRING:     "string",
	types.TNIL:        "nil",
	types.TIDEAL:      "untyped number",
	types.TBLANK:      "blank",
}

var fmtBufferPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

func InstallTypeFormats() {
	types.Sconv = func(s *types.Sym, flag, mode int) string {
		return sconv(s, FmtFlag(flag), FmtMode(mode))
	}
	types.Tconv = func(t *types.Type, flag, mode int) string {
		return tconv(t, FmtFlag(flag), FmtMode(mode))
	}
	types.FormatSym = func(sym *types.Sym, s fmt.State, verb rune, mode int) {
		symFormat(sym, s, verb, FmtMode(mode))
	}
	types.FormatType = func(t *types.Type, s fmt.State, verb rune, mode int) {
		typeFormat(t, s, verb, FmtMode(mode))
	}
}

type fmtType struct {
	x *types.Type
	m FmtMode
}

func (f *fmtType) Format(s fmt.State, verb rune) { typeFormat(f.x, s, verb, f.m) }

// "%L"  print definition, not name
// "%S"  omit 'func' and receiver from function types, short type names
func typeFormat(t *types.Type, s fmt.State, verb rune, mode FmtMode) {
	switch verb {
	case 'v', 'S', 'L':
		if verb == 'v' && s.Flag('+') { // %+v is debug format
			mode = FDbg
		}
		fmt.Fprint(s, tconv(t, fmtFlag(s, verb), mode))
	default:
		fmt.Fprintf(s, "%%!%c(*Type=%p)", verb, t)
	}
}

func tconv(t *types.Type, flag FmtFlag, mode FmtMode) string {
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
func tconv2(b *bytes.Buffer, t *types.Type, flag FmtFlag, mode FmtMode, visited map[*types.Type]int) {
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
	if t.Kind() == types.TSSA {
		b.WriteString(t.Extra.(string))
		return
	}
	if t.Kind() == types.TTUPLE {
		b.WriteString(t.FieldType(0).String())
		b.WriteByte(',')
		b.WriteString(t.FieldType(1).String())
		return
	}

	if t.Kind() == types.TRESULTS {
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
	if t == types.ByteType || t == types.RuneType {
		// in %-T mode collapse rune and byte with their originals.
		switch mode {
		case FTypeIdName, FTypeId:
			t = types.Types[t.Kind()]
		default:
			sconv2(b, t.Sym(), FmtShort, mode)
			return
		}
	}
	if t == types.ErrorType {
		b.WriteString("error")
		return
	}

	// Unless the 'L' flag was specified, if the type has a name, just print that name.
	if flag&FmtLong == 0 && t.Sym() != nil && t != types.Types[t.Kind()] {
		switch mode {
		case FTypeId, FTypeIdName:
			if flag&FmtShort != 0 {
				if t.Vargen != 0 {
					sconv2(b, t.Sym(), FmtShort, mode)
					fmt.Fprintf(b, "·%d", t.Vargen)
					return
				}
				sconv2(b, t.Sym(), FmtShort, mode)
				return
			}

			if mode == FTypeIdName {
				sconv2(b, t.Sym(), FmtUnsigned, mode)
				return
			}

			if t.Sym().Pkg == LocalPkg && t.Vargen != 0 {
				b.WriteString(mode.Sprintf("%v·%d", t.Sym(), t.Vargen))
				return
			}
		}

		sconv2(b, t.Sym(), 0, mode)
		return
	}

	if int(t.Kind()) < len(BasicTypeNames) && BasicTypeNames[t.Kind()] != "" {
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
			name = BasicTypeNames[t.Kind()]
		}
		b.WriteString(name)
		return
	}

	if mode == FDbg {
		b.WriteString(t.Kind().String())
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

	switch t.Kind() {
	case types.TPTR:
		b.WriteByte('*')
		switch mode {
		case FTypeId, FTypeIdName:
			if flag&FmtShort != 0 {
				tconv2(b, t.Elem(), FmtShort, mode, visited)
				return
			}
		}
		tconv2(b, t.Elem(), 0, mode, visited)

	case types.TARRAY:
		b.WriteByte('[')
		b.WriteString(strconv.FormatInt(t.NumElem(), 10))
		b.WriteByte(']')
		tconv2(b, t.Elem(), 0, mode, visited)

	case types.TSLICE:
		b.WriteString("[]")
		tconv2(b, t.Elem(), 0, mode, visited)

	case types.TCHAN:
		switch t.ChanDir() {
		case types.Crecv:
			b.WriteString("<-chan ")
			tconv2(b, t.Elem(), 0, mode, visited)
		case types.Csend:
			b.WriteString("chan<- ")
			tconv2(b, t.Elem(), 0, mode, visited)
		default:
			b.WriteString("chan ")
			if t.Elem() != nil && t.Elem().IsChan() && t.Elem().Sym() == nil && t.Elem().ChanDir() == types.Crecv {
				b.WriteByte('(')
				tconv2(b, t.Elem(), 0, mode, visited)
				b.WriteByte(')')
			} else {
				tconv2(b, t.Elem(), 0, mode, visited)
			}
		}

	case types.TMAP:
		b.WriteString("map[")
		tconv2(b, t.Key(), 0, mode, visited)
		b.WriteByte(']')
		tconv2(b, t.Elem(), 0, mode, visited)

	case types.TINTER:
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

	case types.TFUNC:
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

	case types.TSTRUCT:
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
				base.Fatalf("unknown internal map type")
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

	case types.TFORW:
		b.WriteString("undefined")
		if t.Sym() != nil {
			b.WriteByte(' ')
			sconv2(b, t.Sym(), 0, mode)
		}

	case types.TUNSAFEPTR:
		b.WriteString("unsafe.Pointer")

	case types.Txxx:
		b.WriteString("Txxx")
	default:
		// Don't know how to handle - fall back to detailed prints.
		b.WriteString(mode.Sprintf("%v <%v>", t.Kind(), t.Sym()))
	}
}

func fldconv(b *bytes.Buffer, f *types.Field, flag FmtFlag, mode FmtMode, visited map[*types.Type]int, funarg types.Funarg) {
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
			s = OrigSym(s)
		}

		if s != nil && f.Embedded == 0 {
			if funarg != types.FunargNone {
				name = modeString(AsNode(f.Nname), mode)
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

// Node

func modeString(n Node, mode FmtMode) string { return mode.Sprint(n) }

type fmtNode struct {
	x Node
	m FmtMode
}

func (f *fmtNode) Format(s fmt.State, verb rune) { nodeFormat(f.x, s, verb, f.m) }

func FmtNode(n Node, s fmt.State, verb rune) {
	// %+v prints Dump.
	if s.Flag('+') && verb == 'v' {
		dumpNode(s, n, 1)
		return
	}

	// Otherwise print Go syntax.
	nodeFormat(n, s, verb, FErr)
}

func nodeFormat(n Node, s fmt.State, verb rune, mode FmtMode) {
	switch verb {
	case 'v', 'S', 'L':
		nconvFmt(n, s, fmtFlag(s, verb), mode)

	default:
		fmt.Fprintf(s, "%%!%c(*Node=%p)", verb, n)
	}
}

// "%L"  suffix with "(type %T)" where possible
// "%+S" in debug mode, don't recurse, no multiline output
func nconvFmt(n Node, s fmt.State, flag FmtFlag, mode FmtMode) {
	if n == nil {
		fmt.Fprint(s, "<N>")
		return
	}

	flag, mode = flag.update(mode)

	switch mode {
	case FErr:
		nodeFmt(n, s, flag, mode)

	default:
		base.Fatalf("unhandled %%N mode: %d", mode)
	}
}

func nodeFmt(n Node, s fmt.State, flag FmtFlag, mode FmtMode) {
	t := n.Type()
	if flag&FmtLong != 0 && t != nil {
		if t.Kind() == types.TNIL {
			fmt.Fprint(s, "nil")
		} else if n.Op() == ONAME && n.Name().AutoTemp() {
			mode.Fprintf(s, "%v value", t)
		} else {
			mode.Fprintf(s, "%v (type %v)", n, t)
		}
		return
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if OpPrec[n.Op()] < 0 {
		stmtFmt(n, s, mode)
		return
	}

	exprFmt(n, s, 0, mode)
}

var OpPrec = []int{
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
	OCOMPLIT:       8,
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
	ONIL:           8,
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
	OTSLICE:        8,
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
	OBLOCK:      -1,
	OBREAK:      -1,
	OCASE:       -1,
	OCONTINUE:   -1,
	ODCL:        -1,
	ODEFER:      -1,
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

// Statements which may be rendered with a simplestmt as init.
func StmtWithInit(op Op) bool {
	switch op {
	case OIF, OFOR, OFORUNTIL, OSWITCH:
		return true
	}
	return false
}

func stmtFmt(n Node, s fmt.State, mode FmtMode) {
	// some statements allow for an init, but at most one,
	// but we may have an arbitrary number added, eg by typecheck
	// and inlining. If it doesn't fit the syntax, emit an enclosing
	// block starting with the init statements.

	// if we can just say "for" n->ninit; ... then do so
	simpleinit := n.Init().Len() == 1 && n.Init().First().Init().Len() == 0 && StmtWithInit(n.Op())

	// otherwise, print the inits as separate statements
	complexinit := n.Init().Len() != 0 && !simpleinit && (mode != FErr)

	// but if it was for if/for/switch, put in an extra surrounding block to limit the scope
	extrablock := complexinit && StmtWithInit(n.Op())

	if extrablock {
		fmt.Fprint(s, "{")
	}

	if complexinit {
		mode.Fprintf(s, " %v; ", n.Init())
	}

	switch n.Op() {
	case ODCL:
		mode.Fprintf(s, "var %v %v", n.Left().Sym(), n.Left().Type())

	// Don't export "v = <N>" initializing statements, hope they're always
	// preceded by the DCL which will be re-parsed and typechecked to reproduce
	// the "v = <N>" again.
	case OAS:
		if n.Colas() && !complexinit {
			mode.Fprintf(s, "%v := %v", n.Left(), n.Right())
		} else {
			mode.Fprintf(s, "%v = %v", n.Left(), n.Right())
		}

	case OASOP:
		if n.Implicit() {
			if n.SubOp() == OADD {
				mode.Fprintf(s, "%v++", n.Left())
			} else {
				mode.Fprintf(s, "%v--", n.Left())
			}
			break
		}

		mode.Fprintf(s, "%v %#v= %v", n.Left(), n.SubOp(), n.Right())

	case OAS2, OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		if n.Colas() && !complexinit {
			mode.Fprintf(s, "%.v := %.v", n.List(), n.Rlist())
		} else {
			mode.Fprintf(s, "%.v = %.v", n.List(), n.Rlist())
		}

	case OBLOCK:
		if n.List().Len() != 0 {
			mode.Fprintf(s, "%v", n.List())
		}

	case ORETURN:
		mode.Fprintf(s, "return %.v", n.List())

	case ORETJMP:
		mode.Fprintf(s, "retjmp %v", n.Sym())

	case OINLMARK:
		mode.Fprintf(s, "inlmark %d", n.Offset())

	case OGO:
		mode.Fprintf(s, "go %v", n.Left())

	case ODEFER:
		mode.Fprintf(s, "defer %v", n.Left())

	case OIF:
		if simpleinit {
			mode.Fprintf(s, "if %v; %v { %v }", n.Init().First(), n.Left(), n.Body())
		} else {
			mode.Fprintf(s, "if %v { %v }", n.Left(), n.Body())
		}
		if n.Rlist().Len() != 0 {
			mode.Fprintf(s, " else { %v }", n.Rlist())
		}

	case OFOR, OFORUNTIL:
		opname := "for"
		if n.Op() == OFORUNTIL {
			opname = "foruntil"
		}
		if mode == FErr { // TODO maybe only if FmtShort, same below
			fmt.Fprintf(s, "%s loop", opname)
			break
		}

		fmt.Fprint(s, opname)
		if simpleinit {
			mode.Fprintf(s, " %v;", n.Init().First())
		} else if n.Right() != nil {
			fmt.Fprint(s, " ;")
		}

		if n.Left() != nil {
			mode.Fprintf(s, " %v", n.Left())
		}

		if n.Right() != nil {
			mode.Fprintf(s, "; %v", n.Right())
		} else if simpleinit {
			fmt.Fprint(s, ";")
		}

		if n.Op() == OFORUNTIL && n.List().Len() != 0 {
			mode.Fprintf(s, "; %v", n.List())
		}

		mode.Fprintf(s, " { %v }", n.Body())

	case ORANGE:
		if mode == FErr {
			fmt.Fprint(s, "for loop")
			break
		}

		if n.List().Len() == 0 {
			mode.Fprintf(s, "for range %v { %v }", n.Right(), n.Body())
			break
		}

		mode.Fprintf(s, "for %.v = range %v { %v }", n.List(), n.Right(), n.Body())

	case OSELECT, OSWITCH:
		if mode == FErr {
			mode.Fprintf(s, "%v statement", n.Op())
			break
		}

		mode.Fprintf(s, "%#v", n.Op())
		if simpleinit {
			mode.Fprintf(s, " %v;", n.Init().First())
		}
		if n.Left() != nil {
			mode.Fprintf(s, " %v ", n.Left())
		}

		mode.Fprintf(s, " { %v }", n.List())

	case OCASE:
		if n.List().Len() != 0 {
			mode.Fprintf(s, "case %.v", n.List())
		} else {
			fmt.Fprint(s, "default")
		}
		mode.Fprintf(s, ": %v", n.Body())

	case OBREAK, OCONTINUE, OGOTO, OFALL:
		if n.Sym() != nil {
			mode.Fprintf(s, "%#v %v", n.Op(), n.Sym())
		} else {
			mode.Fprintf(s, "%#v", n.Op())
		}

	case OLABEL:
		mode.Fprintf(s, "%v: ", n.Sym())
	}

	if extrablock {
		fmt.Fprint(s, "}")
	}
}

func exprFmt(n Node, s fmt.State, prec int, mode FmtMode) {
	for {
		if n == nil {
			fmt.Fprint(s, "<N>")
			return
		}

		// We always want the original, if any.
		if o := Orig(n); o != n {
			n = o
			continue
		}

		// Skip implicit operations introduced during typechecking.
		switch n.Op() {
		case OADDR, ODEREF, OCONV, OCONVNOP, OCONVIFACE:
			if n.Implicit() {
				n = n.Left()
				continue
			}
		}

		break
	}

	nprec := OpPrec[n.Op()]
	if n.Op() == OTYPE && n.Sym() != nil {
		nprec = 8
	}

	if prec > nprec {
		mode.Fprintf(s, "(%v)", n)
		return
	}

	switch n.Op() {
	case OPAREN:
		mode.Fprintf(s, "(%v)", n.Left())

	case ONIL:
		fmt.Fprint(s, "nil")

	case OLITERAL: // this is a bit of a mess
		if mode == FErr && n.Sym() != nil {
			fmt.Fprint(s, smodeString(n.Sym(), mode))
			return
		}

		needUnparen := false
		if n.Type() != nil && !n.Type().IsUntyped() {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if n.Type().IsPtr() || (n.Type().IsChan() && n.Type().ChanDir() == types.Crecv) {
				mode.Fprintf(s, "(%v)(", n.Type())
			} else {
				mode.Fprintf(s, "%v(", n.Type())
			}
			needUnparen = true
		}

		if n.Type() == types.UntypedRune {
			switch x, ok := constant.Int64Val(n.Val()); {
			case !ok:
				fallthrough
			default:
				fmt.Fprintf(s, "('\\x00' + %v)", n.Val())

			case ' ' <= x && x < utf8.RuneSelf && x != '\\' && x != '\'':
				fmt.Fprintf(s, "'%c'", int(x))

			case 0 <= x && x < 1<<16:
				fmt.Fprintf(s, "'\\u%04x'", uint(int(x)))

			case 0 <= x && x <= utf8.MaxRune:
				fmt.Fprintf(s, "'\\U%08x'", uint64(x))
			}
		} else {
			fmt.Fprint(s, FmtConst(n.Val(), fmtFlag(s, 'v')))
		}

		if needUnparen {
			mode.Fprintf(s, ")")
		}

	case ODCLFUNC:
		if sym := n.Sym(); sym != nil {
			fmt.Fprint(s, smodeString(sym, mode))
			return
		}
		mode.Fprintf(s, "<unnamed Func>")

	case ONAME:
		// Special case: name used as local variable in export.
		// _ becomes ~b%d internally; print as _ for export
		if mode == FErr && n.Sym() != nil && n.Sym().Name[0] == '~' && n.Sym().Name[1] == 'b' {
			fmt.Fprint(s, "_")
			return
		}
		fallthrough
	case OPACK, ONONAME, OMETHEXPR:
		fmt.Fprint(s, smodeString(n.Sym(), mode))

	case OTYPE:
		if n.Type() == nil && n.Sym() != nil {
			fmt.Fprint(s, smodeString(n.Sym(), mode))
			return
		}
		mode.Fprintf(s, "%v", n.Type())

	case OTSLICE:
		n := n.(*SliceType)
		if n.DDD {
			mode.Fprintf(s, "...%v", n.Elem)
		} else {
			mode.Fprintf(s, "[]%v", n.Elem) // happens before typecheck
		}

	case OTARRAY:
		n := n.(*ArrayType)
		if n.Len == nil {
			mode.Fprintf(s, "[...]%v", n.Elem)
		} else {
			mode.Fprintf(s, "[%v]%v", n.Len, n.Elem)
		}

	case OTMAP:
		n := n.(*MapType)
		mode.Fprintf(s, "map[%v]%v", n.Key, n.Elem)

	case OTCHAN:
		n := n.(*ChanType)
		switch n.Dir {
		case types.Crecv:
			mode.Fprintf(s, "<-chan %v", n.Elem)

		case types.Csend:
			mode.Fprintf(s, "chan<- %v", n.Elem)

		default:
			if n.Elem != nil && n.Elem.Op() == OTCHAN && n.Elem.(*ChanType).Dir == types.Crecv {
				mode.Fprintf(s, "chan (%v)", n.Elem)
			} else {
				mode.Fprintf(s, "chan %v", n.Elem)
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
		if n.Body().Len() != 0 {
			mode.Fprintf(s, "%v { %v }", n.Type(), n.Body())
			return
		}
		mode.Fprintf(s, "%v { %v }", n.Type(), n.Func().Body())

	case OCOMPLIT:
		if mode == FErr {
			if n.Implicit() {
				mode.Fprintf(s, "... argument")
				return
			}
			if n.Right() != nil {
				mode.Fprintf(s, "%v{%s}", n.Right(), ellipsisIf(n.List().Len() != 0))
				return
			}

			fmt.Fprint(s, "composite literal")
			return
		}
		mode.Fprintf(s, "(%v{ %.v })", n.Right(), n.List())

	case OPTRLIT:
		mode.Fprintf(s, "&%v", n.Left())

	case OSTRUCTLIT, OARRAYLIT, OSLICELIT, OMAPLIT:
		if mode == FErr {
			mode.Fprintf(s, "%v{%s}", n.Type(), ellipsisIf(n.List().Len() != 0))
			return
		}
		mode.Fprintf(s, "(%v{ %.v })", n.Type(), n.List())

	case OKEY:
		if n.Left() != nil && n.Right() != nil {
			mode.Fprintf(s, "%v:%v", n.Left(), n.Right())
			return
		}

		if n.Left() == nil && n.Right() != nil {
			mode.Fprintf(s, ":%v", n.Right())
			return
		}
		if n.Left() != nil && n.Right() == nil {
			mode.Fprintf(s, "%v:", n.Left())
			return
		}
		fmt.Fprint(s, ":")

	case OSTRUCTKEY:
		mode.Fprintf(s, "%v:%v", n.Sym(), n.Left())

	case OCALLPART:
		exprFmt(n.Left(), s, nprec, mode)
		if n.Sym() == nil {
			fmt.Fprint(s, ".<nil>")
			return
		}
		mode.Fprintf(s, ".%0S", n.Sym())

	case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
		exprFmt(n.Left(), s, nprec, mode)
		if n.Sym() == nil {
			fmt.Fprint(s, ".<nil>")
			return
		}
		mode.Fprintf(s, ".%0S", n.Sym())

	case ODOTTYPE, ODOTTYPE2:
		exprFmt(n.Left(), s, nprec, mode)
		if n.Right() != nil {
			mode.Fprintf(s, ".(%v)", n.Right())
			return
		}
		mode.Fprintf(s, ".(%v)", n.Type())

	case OINDEX, OINDEXMAP:
		exprFmt(n.Left(), s, nprec, mode)
		mode.Fprintf(s, "[%v]", n.Right())

	case OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
		exprFmt(n.Left(), s, nprec, mode)
		fmt.Fprint(s, "[")
		low, high, max := n.SliceBounds()
		if low != nil {
			fmt.Fprint(s, modeString(low, mode))
		}
		fmt.Fprint(s, ":")
		if high != nil {
			fmt.Fprint(s, modeString(high, mode))
		}
		if n.Op().IsSlice3() {
			fmt.Fprint(s, ":")
			if max != nil {
				fmt.Fprint(s, modeString(max, mode))
			}
		}
		fmt.Fprint(s, "]")

	case OSLICEHEADER:
		if n.List().Len() != 2 {
			base.Fatalf("bad OSLICEHEADER list length %d", n.List().Len())
		}
		mode.Fprintf(s, "sliceheader{%v,%v,%v}", n.Left(), n.List().First(), n.List().Second())

	case OCOMPLEX, OCOPY:
		if n.Left() != nil {
			mode.Fprintf(s, "%#v(%v, %v)", n.Op(), n.Left(), n.Right())
		} else {
			mode.Fprintf(s, "%#v(%.v)", n.Op(), n.List())
		}

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OBYTES2STR,
		ORUNES2STR,
		OSTR2BYTES,
		OSTR2RUNES,
		ORUNESTR:
		if n.Type() == nil || n.Type().Sym() == nil {
			mode.Fprintf(s, "(%v)", n.Type())
		} else {
			mode.Fprintf(s, "%v", n.Type())
		}
		if n.Left() != nil {
			mode.Fprintf(s, "(%v)", n.Left())
		} else {
			mode.Fprintf(s, "(%.v)", n.List())
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
		if n.Left() != nil {
			mode.Fprintf(s, "%#v(%v)", n.Op(), n.Left())
			return
		}
		if n.IsDDD() {
			mode.Fprintf(s, "%#v(%.v...)", n.Op(), n.List())
			return
		}
		mode.Fprintf(s, "%#v(%.v)", n.Op(), n.List())

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		exprFmt(n.Left(), s, nprec, mode)
		if n.IsDDD() {
			mode.Fprintf(s, "(%.v...)", n.List())
			return
		}
		mode.Fprintf(s, "(%.v)", n.List())

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		if n.List().Len() != 0 { // pre-typecheck
			mode.Fprintf(s, "make(%v, %.v)", n.Type(), n.List())
			return
		}
		if n.Right() != nil {
			mode.Fprintf(s, "make(%v, %v, %v)", n.Type(), n.Left(), n.Right())
			return
		}
		if n.Left() != nil && (n.Op() == OMAKESLICE || !n.Left().Type().IsUntyped()) {
			mode.Fprintf(s, "make(%v, %v)", n.Type(), n.Left())
			return
		}
		mode.Fprintf(s, "make(%v)", n.Type())

	case OMAKESLICECOPY:
		mode.Fprintf(s, "makeslicecopy(%v, %v, %v)", n.Type(), n.Left(), n.Right())

	case OPLUS, ONEG, OADDR, OBITNOT, ODEREF, ONOT, ORECV:
		// Unary
		mode.Fprintf(s, "%#v", n.Op())
		if n.Left() != nil && n.Left().Op() == n.Op() {
			fmt.Fprint(s, " ")
		}
		exprFmt(n.Left(), s, nprec+1, mode)

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
		exprFmt(n.Left(), s, nprec, mode)
		mode.Fprintf(s, " %#v ", n.Op())
		exprFmt(n.Right(), s, nprec+1, mode)

	case OADDSTR:
		for i, n1 := range n.List().Slice() {
			if i != 0 {
				fmt.Fprint(s, " + ")
			}
			exprFmt(n1, s, nprec, mode)
		}
	default:
		mode.Fprintf(s, "<node %v>", n.Op())
	}
}

func ellipsisIf(b bool) string {
	if b {
		return "..."
	}
	return ""
}

// Nodes

type fmtNodes struct {
	x Nodes
	m FmtMode
}

func (f *fmtNodes) Format(s fmt.State, verb rune) { f.x.format(s, verb, FErr) }

func (l Nodes) Format(s fmt.State, verb rune) { l.format(s, verb, FErr) }

func (l Nodes) format(s fmt.State, verb rune, mode FmtMode) {
	if s.Flag('+') && verb == 'v' {
		// %+v is DumpList output
		dumpNodes(s, l, 1)
		return
	}

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
func (l Nodes) hconv(s fmt.State, flag FmtFlag, mode FmtMode) {
	flag, mode = flag.update(mode)
	sep := "; "
	if flag&FmtComma != 0 {
		sep = ", "
	}

	for i, n := range l.Slice() {
		fmt.Fprint(s, modeString(n, mode))
		if i+1 < l.Len() {
			fmt.Fprint(s, sep)
		}
	}
}

// Dump

func Dump(s string, n Node) {
	fmt.Printf("%s [%p]%+v", s, n, n)
}

func DumpList(s string, l Nodes) {
	var buf bytes.Buffer
	FDumpList(&buf, s, l)
	os.Stdout.Write(buf.Bytes())
}

func FDumpList(w io.Writer, s string, l Nodes) {
	io.WriteString(w, s)
	dumpNodes(w, l, 1)
	io.WriteString(w, "\n")
}

// indent prints indentation to w.
func indent(w io.Writer, depth int) {
	fmt.Fprint(w, "\n")
	for i := 0; i < depth; i++ {
		fmt.Fprint(w, ".   ")
	}
}

// EscFmt is set by the escape analysis code to add escape analysis details to the node print.
var EscFmt func(n Node) string

// dumpNodeHeader prints the debug-format node header line to w.
func dumpNodeHeader(w io.Writer, n Node) {
	// Useful to see which nodes in an AST printout are actually identical
	if base.Debug.DumpPtrs != 0 {
		fmt.Fprintf(w, " p(%p)", n)
	}
	if n.Name() != nil && n.Name().Vargen != 0 {
		fmt.Fprintf(w, " g(%d)", n.Name().Vargen)
	}

	if base.Debug.DumpPtrs != 0 && n.Name() != nil && n.Name().Defn != nil {
		// Useful to see where Defn is set and what node it points to
		fmt.Fprintf(w, " defn(%p)", n.Name().Defn)
	}

	if n.Pos().IsKnown() {
		pfx := ""
		switch n.Pos().IsStmt() {
		case src.PosNotStmt:
			pfx = "_" // "-" would be confusing
		case src.PosIsStmt:
			pfx = "+"
		}
		fmt.Fprintf(w, " l(%s%d)", pfx, n.Pos().Line())
	}

	if n.Offset() != types.BADWIDTH {
		fmt.Fprintf(w, " x(%d)", n.Offset())
	}

	if n.Class() != 0 {
		fmt.Fprintf(w, " class(%v)", n.Class())
	}

	if n.Colas() {
		fmt.Fprintf(w, " colas(%v)", n.Colas())
	}

	if EscFmt != nil {
		if esc := EscFmt(n); esc != "" {
			fmt.Fprintf(w, " %s", esc)
		}
	}

	if n.Typecheck() != 0 {
		fmt.Fprintf(w, " tc(%d)", n.Typecheck())
	}

	if n.IsDDD() {
		fmt.Fprintf(w, " isddd(%v)", n.IsDDD())
	}

	if n.Implicit() {
		fmt.Fprintf(w, " implicit(%v)", n.Implicit())
	}

	if n.Op() == ONAME {
		if n.Name().Addrtaken() {
			fmt.Fprint(w, " addrtaken")
		}
		if n.Name().Assigned() {
			fmt.Fprint(w, " assigned")
		}
		if n.Name().IsClosureVar() {
			fmt.Fprint(w, " closurevar")
		}
		if n.Name().Captured() {
			fmt.Fprint(w, " captured")
		}
		if n.Name().IsOutputParamHeapAddr() {
			fmt.Fprint(w, " outputparamheapaddr")
		}
	}
	if n.Bounded() {
		fmt.Fprint(w, " bounded")
	}
	if n.NonNil() {
		fmt.Fprint(w, " nonnil")
	}

	if n.HasCall() {
		fmt.Fprint(w, " hascall")
	}

	if n.Name() != nil && n.Name().Used() {
		fmt.Fprint(w, " used")
	}
}

func dumpNode(w io.Writer, n Node, depth int) {
	indent(w, depth)
	if depth > 40 {
		fmt.Fprint(w, "...")
		return
	}

	if n.Init().Len() != 0 {
		fmt.Fprintf(w, "%+v-init", n.Op())
		dumpNodes(w, n.Init(), depth+1)
		indent(w, depth)
	}

	switch n.Op() {
	default:
		fmt.Fprintf(w, "%+v", n.Op())
		dumpNodeHeader(w, n)

	case OLITERAL:
		fmt.Fprintf(w, "%+v-%v", n.Op(), n.Val())
		dumpNodeHeader(w, n)

	case ONAME, ONONAME, OMETHEXPR:
		if n.Sym() != nil {
			fmt.Fprintf(w, "%+v-%+v", n.Op(), n.Sym())
		} else {
			fmt.Fprintf(w, "%+v", n.Op())
		}
		dumpNodeHeader(w, n)
		if n.Type() == nil && n.Name() != nil && n.Name().Ntype != nil {
			indent(w, depth)
			fmt.Fprintf(w, "%+v-ntype", n.Op())
			dumpNode(w, n.Name().Ntype, depth+1)
		}

	case OASOP:
		fmt.Fprintf(w, "%+v-%+v", n.Op(), n.SubOp())
		dumpNodeHeader(w, n)

	case OTYPE:
		fmt.Fprintf(w, "%+v %+v", n.Op(), n.Sym())
		dumpNodeHeader(w, n)
		fmt.Fprintf(w, " type=%+v", n.Type())
		if n.Type() == nil && n.Name() != nil && n.Name().Ntype != nil {
			indent(w, depth)
			fmt.Fprintf(w, "%+v-ntype", n.Op())
			dumpNode(w, n.Name().Ntype, depth+1)
		}
	}

	if n.Op() == OCLOSURE && n.Func() != nil && n.Func().Nname.Sym() != nil {
		fmt.Fprintf(w, " fnName %+v", n.Func().Nname.Sym())
	}
	if n.Sym() != nil && n.Op() != ONAME {
		fmt.Fprintf(w, " %+v", n.Sym())
	}

	if n.Type() != nil {
		fmt.Fprintf(w, " %+v", n.Type())
	}

	if n.Left() != nil {
		dumpNode(w, n.Left(), depth+1)
	}
	if n.Right() != nil {
		dumpNode(w, n.Right(), depth+1)
	}
	if n.Op() == OCLOSURE && n.Func() != nil && n.Func().Body().Len() != 0 {
		indent(w, depth)
		// The function associated with a closure
		fmt.Fprintf(w, "%+v-clofunc", n.Op())
		dumpNode(w, n.Func(), depth+1)
	}
	if n.Op() == ODCLFUNC && n.Func() != nil && n.Func().Dcl != nil && len(n.Func().Dcl) != 0 {
		indent(w, depth)
		// The dcls for a func or closure
		fmt.Fprintf(w, "%+v-dcl", n.Op())
		for _, dcl := range n.Func().Dcl {
			dumpNode(w, dcl, depth+1)
		}
	}
	if n.List().Len() != 0 {
		indent(w, depth)
		fmt.Fprintf(w, "%+v-list", n.Op())
		dumpNodes(w, n.List(), depth+1)
	}

	if n.Rlist().Len() != 0 {
		indent(w, depth)
		fmt.Fprintf(w, "%+v-rlist", n.Op())
		dumpNodes(w, n.Rlist(), depth+1)
	}

	if n.Body().Len() != 0 {
		indent(w, depth)
		fmt.Fprintf(w, "%+v-body", n.Op())
		dumpNodes(w, n.Body(), depth+1)
	}
}

func dumpNodes(w io.Writer, list Nodes, depth int) {
	if list.Len() == 0 {
		fmt.Fprintf(w, " <nil>")
		return
	}

	for _, n := range list.Slice() {
		dumpNode(w, n, depth)
	}
}
