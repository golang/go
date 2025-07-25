// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strconv"
	"sync"

	"cmd/compile/internal/base"
	"cmd/internal/hash"
)

// BuiltinPkg is a fake package that declares the universe block.
var BuiltinPkg *Pkg

// LocalPkg is the package being compiled.
var LocalPkg *Pkg

// UnsafePkg is package unsafe.
var UnsafePkg *Pkg

// BlankSym is the blank (_) symbol.
var BlankSym *Sym

// numImport tracks how often a package with a given name is imported.
// It is used to provide a better error message (by using the package
// path to disambiguate) if a package that appears multiple times with
// the same name appears in an error message.
var NumImport = make(map[string]int)

// fmtMode represents the kind of printing being done.
// The default is regular Go syntax (fmtGo).
// fmtDebug is like fmtGo but for debugging dumps and prints the type kind too.
// fmtTypeID and fmtTypeIDName are for generating various unique representations
// of types used in hashes, the linker, and function/method instantiations.
type fmtMode int

const (
	fmtGo fmtMode = iota
	fmtDebug
	fmtTypeID
	fmtTypeIDName
)

// Sym

// Format implements formatting for a Sym.
// The valid formats are:
//
//	%v	Go syntax: Name for symbols in the local package, PkgName.Name for imported symbols.
//	%+v	Debug syntax: always include PkgName. prefix even for local names.
//	%S	Short syntax: Name only, no matter what.
func (s *Sym) Format(f fmt.State, verb rune) {
	mode := fmtGo
	switch verb {
	case 'v', 'S':
		if verb == 'v' && f.Flag('+') {
			mode = fmtDebug
		}
		fmt.Fprint(f, sconv(s, verb, mode))

	default:
		fmt.Fprintf(f, "%%!%c(*types.Sym=%p)", verb, s)
	}
}

func (s *Sym) String() string {
	return sconv(s, 0, fmtGo)
}

// See #16897 for details about performance implications
// before changing the implementation of sconv.
func sconv(s *Sym, verb rune, mode fmtMode) string {
	if verb == 'L' {
		panic("linksymfmt")
	}

	if s == nil {
		return "<S>"
	}

	q := pkgqual(s.Pkg, verb, mode)
	if q == "" {
		return s.Name
	}

	buf := fmtBufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer fmtBufferPool.Put(buf)

	buf.WriteString(q)
	buf.WriteByte('.')
	buf.WriteString(s.Name)
	return InternString(buf.Bytes())
}

func sconv2(b *bytes.Buffer, s *Sym, verb rune, mode fmtMode) {
	if verb == 'L' {
		panic("linksymfmt")
	}
	if s == nil {
		b.WriteString("<S>")
		return
	}

	symfmt(b, s, verb, mode)
}

func symfmt(b *bytes.Buffer, s *Sym, verb rune, mode fmtMode) {
	name := s.Name
	if q := pkgqual(s.Pkg, verb, mode); q != "" {
		b.WriteString(q)
		b.WriteByte('.')
	}
	b.WriteString(name)
}

// pkgqual returns the qualifier that should be used for printing
// symbols from the given package in the given mode.
// If it returns the empty string, no qualification is needed.
func pkgqual(pkg *Pkg, verb rune, mode fmtMode) string {
	if pkg == nil {
		return ""
	}
	if verb != 'S' {
		switch mode {
		case fmtGo: // This is for the user
			if pkg == BuiltinPkg || pkg == LocalPkg {
				return ""
			}

			// If the name was used by multiple packages, display the full path,
			if pkg.Name != "" && NumImport[pkg.Name] > 1 {
				return strconv.Quote(pkg.Path)
			}
			return pkg.Name

		case fmtDebug:
			return pkg.Name

		case fmtTypeIDName:
			// dcommontype, typehash
			return pkg.Name

		case fmtTypeID:
			// (methodsym), typesym, weaksym
			return pkg.Prefix
		}
	}

	return ""
}

// Type

var BasicTypeNames = []string{
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

// Format implements formatting for a Type.
// The valid formats are:
//
//	%v	Go syntax
//	%+v	Debug syntax: Go syntax with a KIND- prefix for all but builtins.
//	%L	Go syntax for underlying type if t is named
//	%S	short Go syntax: drop leading "func" in function type
//	%-S	special case for method receiver symbol
func (t *Type) Format(s fmt.State, verb rune) {
	mode := fmtGo
	switch verb {
	case 'v', 'S', 'L':
		if verb == 'v' && s.Flag('+') { // %+v is debug format
			mode = fmtDebug
		}
		if verb == 'S' && s.Flag('-') { // %-S is special case for receiver - short typeid format
			mode = fmtTypeID
		}
		fmt.Fprint(s, tconv(t, verb, mode))
	default:
		fmt.Fprintf(s, "%%!%c(*Type=%p)", verb, t)
	}
}

// String returns the Go syntax for the type t.
func (t *Type) String() string {
	return tconv(t, 0, fmtGo)
}

// LinkString returns a string description of t, suitable for use in
// link symbols.
//
// The description corresponds to type identity. That is, for any pair
// of types t1 and t2, Identical(t1, t2) == (t1.LinkString() ==
// t2.LinkString()) is true. Thus it's safe to use as a map key to
// implement a type-identity-keyed map.
func (t *Type) LinkString() string {
	return tconv(t, 0, fmtTypeID)
}

// NameString generates a user-readable, mostly unique string
// description of t. NameString always returns the same description
// for identical types, even across compilation units.
//
// NameString qualifies identifiers by package name, so it has
// collisions when different packages share the same names and
// identifiers. It also does not distinguish function-scope defined
// types from package-scoped defined types or from each other.
func (t *Type) NameString() string {
	return tconv(t, 0, fmtTypeIDName)
}

func tconv(t *Type, verb rune, mode fmtMode) string {
	buf := fmtBufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer fmtBufferPool.Put(buf)

	tconv2(buf, t, verb, mode, nil)
	return InternString(buf.Bytes())
}

// tconv2 writes a string representation of t to b.
// flag and mode control exactly what is printed.
// Any types x that are already in the visited map get printed as @%d where %d=visited[x].
// See #16897 before changing the implementation of tconv.
func tconv2(b *bytes.Buffer, t *Type, verb rune, mode fmtMode, visited map[*Type]int) {
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
	if t.Kind() == TSSA {
		b.WriteString(t.extra.(string))
		return
	}
	if t.Kind() == TTUPLE {
		b.WriteString(t.FieldType(0).String())
		b.WriteByte(',')
		b.WriteString(t.FieldType(1).String())
		return
	}

	if t.Kind() == TRESULTS {
		tys := t.extra.(*Results).Types
		for i, et := range tys {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(et.String())
		}
		return
	}

	if t == AnyType || t == ByteType || t == RuneType {
		// in %-T mode collapse predeclared aliases with their originals.
		switch mode {
		case fmtTypeIDName, fmtTypeID:
			t = Types[t.Kind()]
		default:
			sconv2(b, t.Sym(), 'S', mode)
			return
		}
	}
	if t == ErrorType {
		b.WriteString("error")
		return
	}

	// Unless the 'L' flag was specified, if the type has a name, just print that name.
	if verb != 'L' && t.Sym() != nil && t != Types[t.Kind()] {
		// Default to 'v' if verb is invalid.
		if verb != 'S' {
			verb = 'v'
		}

		// In unified IR, function-scope defined types will have a ·N
		// suffix embedded directly in their Name. Trim this off for
		// non-fmtTypeID modes.
		sym := t.Sym()
		if mode != fmtTypeID {
			base, _ := SplitVargenSuffix(sym.Name)
			if len(base) < len(sym.Name) {
				sym = &Sym{Pkg: sym.Pkg, Name: base}
			}
		}
		sconv2(b, sym, verb, mode)
		return
	}

	if int(t.Kind()) < len(BasicTypeNames) && BasicTypeNames[t.Kind()] != "" {
		var name string
		switch t {
		case UntypedBool:
			name = "untyped bool"
		case UntypedString:
			name = "untyped string"
		case UntypedInt:
			name = "untyped int"
		case UntypedRune:
			name = "untyped rune"
		case UntypedFloat:
			name = "untyped float"
		case UntypedComplex:
			name = "untyped complex"
		default:
			name = BasicTypeNames[t.Kind()]
		}
		b.WriteString(name)
		return
	}

	if mode == fmtDebug {
		b.WriteString(t.Kind().String())
		b.WriteByte('-')
		tconv2(b, t, 'v', fmtGo, visited)
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
		visited = map[*Type]int{}
	}
	visited[t] = b.Len()
	defer delete(visited, t)

	switch t.Kind() {
	case TPTR:
		b.WriteByte('*')
		switch mode {
		case fmtTypeID, fmtTypeIDName:
			if verb == 'S' {
				tconv2(b, t.Elem(), 'S', mode, visited)
				return
			}
		}
		tconv2(b, t.Elem(), 'v', mode, visited)

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
		case Crecv:
			b.WriteString("<-chan ")
			tconv2(b, t.Elem(), 0, mode, visited)
		case Csend:
			b.WriteString("chan<- ")
			tconv2(b, t.Elem(), 0, mode, visited)
		default:
			b.WriteString("chan ")
			if t.Elem() != nil && t.Elem().IsChan() && t.Elem().Sym() == nil && t.Elem().ChanDir() == Crecv {
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
		for i, f := range t.AllMethods() {
			if i != 0 {
				b.WriteByte(';')
			}
			b.WriteByte(' ')
			switch {
			case f.Sym == nil:
				// Check first that a symbol is defined for this type.
				// Wrong interface definitions may have types lacking a symbol.
				break
			case IsExported(f.Sym.Name):
				sconv2(b, f.Sym, 'S', mode)
			default:
				if mode != fmtTypeIDName {
					mode = fmtTypeID
				}
				sconv2(b, f.Sym, 'v', mode)
			}
			tconv2(b, f.Type, 'S', mode, visited)
		}
		if len(t.AllMethods()) != 0 {
			b.WriteByte(' ')
		}
		b.WriteByte('}')

	case TFUNC:
		if verb == 'S' {
			// no leading func
		} else {
			if t.Recv() != nil {
				b.WriteString("method")
				formatParams(b, t.Recvs(), mode, visited)
				b.WriteByte(' ')
			}
			b.WriteString("func")
		}
		formatParams(b, t.Params(), mode, visited)

		switch t.NumResults() {
		case 0:
			// nothing to do

		case 1:
			b.WriteByte(' ')
			tconv2(b, t.Result(0).Type, 0, mode, visited) // struct->field->field's type

		default:
			b.WriteByte(' ')
			formatParams(b, t.Results(), mode, visited)
		}

	case TSTRUCT:
		if m := t.StructType().Map; m != nil {
			mt := m.MapType()
			// Format the bucket struct for map[x]y as map.group[x]y.
			// This avoids a recursive print that generates very long names.
			switch t {
			case mt.SwissGroup:
				b.WriteString("map.group[")
			default:
				base.Fatalf("unknown internal map type")
			}
			tconv2(b, m.Key(), 0, mode, visited)
			b.WriteByte(']')
			tconv2(b, m.Elem(), 0, mode, visited)
			break
		}

		b.WriteString("struct {")
		for i, f := range t.Fields() {
			if i != 0 {
				b.WriteByte(';')
			}
			b.WriteByte(' ')
			fldconv(b, f, 'L', mode, visited, false)
		}
		if t.NumFields() != 0 {
			b.WriteByte(' ')
		}
		b.WriteByte('}')

	case TFORW:
		b.WriteString("undefined")
		if t.Sym() != nil {
			b.WriteByte(' ')
			sconv2(b, t.Sym(), 'v', mode)
		}

	case TUNSAFEPTR:
		b.WriteString("unsafe.Pointer")

	case Txxx:
		b.WriteString("Txxx")

	default:
		// Don't know how to handle - fall back to detailed prints
		b.WriteString(t.Kind().String())
		b.WriteString(" <")
		sconv2(b, t.Sym(), 'v', mode)
		b.WriteString(">")

	}
}

func formatParams(b *bytes.Buffer, params []*Field, mode fmtMode, visited map[*Type]int) {
	b.WriteByte('(')
	fieldVerb := 'v'
	switch mode {
	case fmtTypeID, fmtTypeIDName, fmtGo:
		// no argument names on function signature, and no "noescape"/"nosplit" tags
		fieldVerb = 'S'
	}
	for i, param := range params {
		if i != 0 {
			b.WriteString(", ")
		}
		fldconv(b, param, fieldVerb, mode, visited, true)
	}
	b.WriteByte(')')
}

func fldconv(b *bytes.Buffer, f *Field, verb rune, mode fmtMode, visited map[*Type]int, isParam bool) {
	if f == nil {
		b.WriteString("<T>")
		return
	}

	var name string
	nameSep := " "
	if verb != 'S' {
		s := f.Sym

		// Using type aliases and embedded fields, it's possible to
		// construct types that can't be directly represented as a
		// type literal. For example, given "type Int = int" (#50190),
		// it would be incorrect to format "struct{ Int }" as either
		// "struct{ int }" or "struct{ Int int }", because those each
		// represent other, distinct types.
		//
		// So for the purpose of LinkString (i.e., fmtTypeID), we use
		// the non-standard syntax "struct{ Int = int }" to represent
		// embedded fields that have been renamed through the use of
		// type aliases.
		if f.Embedded != 0 {
			if mode == fmtTypeID {
				nameSep = " = "

				// Compute tsym, the symbol that would normally be used as
				// the field name when embedding f.Type.
				// TODO(mdempsky): Check for other occurrences of this logic
				// and deduplicate.
				typ := f.Type
				if typ.IsPtr() {
					base.Assertf(typ.Sym() == nil, "embedded pointer type has name: %L", typ)
					typ = typ.Elem()
				}
				tsym := typ.Sym()

				// If the field name matches the embedded type's name, then
				// suppress printing of the field name. For example, format
				// "struct{ T }" as simply that instead of "struct{ T = T }".
				if tsym != nil && (s == tsym || IsExported(tsym.Name) && s.Name == tsym.Name) {
					s = nil
				}
			} else {
				// Suppress the field name for embedded fields for
				// non-LinkString formats, to match historical behavior.
				// TODO(mdempsky): Re-evaluate this.
				s = nil
			}
		}

		if s != nil {
			if isParam {
				name = fmt.Sprint(f.Nname)
			} else if verb == 'L' {
				name = s.Name
				if !IsExported(name) && mode != fmtTypeIDName {
					name = sconv(s, 0, mode) // qualify non-exported names (used on structs, not on funarg)
				}
			} else {
				name = sconv(s, 0, mode)
			}
		}
	}

	if name != "" {
		b.WriteString(name)
		b.WriteString(nameSep)
	}

	if f.IsDDD() {
		var et *Type
		if f.Type != nil {
			et = f.Type.Elem()
		}
		b.WriteString("...")
		tconv2(b, et, 0, mode, visited)
	} else {
		tconv2(b, f.Type, 0, mode, visited)
	}

	if verb != 'S' && !isParam && f.Note != "" {
		b.WriteString(" ")
		b.WriteString(strconv.Quote(f.Note))
	}
}

// SplitVargenSuffix returns name split into a base string and a ·N
// suffix, if any.
func SplitVargenSuffix(name string) (base, suffix string) {
	i := len(name)
	for i > 0 && name[i-1] >= '0' && name[i-1] <= '9' {
		i--
	}
	const dot = "·"
	if i >= len(dot) && name[i-len(dot):i] == dot {
		i -= len(dot)
		return name[:i], name[i:]
	}
	return name, ""
}

// TypeHash computes a hash value for type t to use in type switch statements.
func TypeHash(t *Type) uint32 {
	p := t.LinkString()

	// Using a cryptographic hash is overkill but minimizes accidental collisions.
	h := hash.Sum32([]byte(p))
	return binary.LittleEndian.Uint32(h[:4])
}
