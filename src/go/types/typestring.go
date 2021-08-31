// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of types.

package types

import (
	"bytes"
	"fmt"
	"go/token"
	"unicode/utf8"
)

// A Qualifier controls how named package-level objects are printed in
// calls to TypeString, ObjectString, and SelectionString.
//
// These three formatting routines call the Qualifier for each
// package-level object O, and if the Qualifier returns a non-empty
// string p, the object is printed in the form p.O.
// If it returns an empty string, only the object name O is printed.
//
// Using a nil Qualifier is equivalent to using (*Package).Path: the
// object is qualified by the import path, e.g., "encoding/json.Marshal".
//
type Qualifier func(*Package) string

// RelativeTo returns a Qualifier that fully qualifies members of
// all packages other than pkg.
func RelativeTo(pkg *Package) Qualifier {
	if pkg == nil {
		return nil
	}
	return func(other *Package) string {
		if pkg == other {
			return "" // same package; unqualified
		}
		return other.Path()
	}
}

// TypeString returns the string representation of typ.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
func TypeString(typ Type, qf Qualifier) string {
	var buf bytes.Buffer
	WriteType(&buf, typ, qf)
	return buf.String()
}

// WriteType writes the string representation of typ to buf.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
func WriteType(buf *bytes.Buffer, typ Type, qf Qualifier) {
	newTypeWriter(buf, qf).typ(typ)
}

// WriteSignature writes the representation of the signature sig to buf,
// without a leading "func" keyword.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
func WriteSignature(buf *bytes.Buffer, sig *Signature, qf Qualifier) {
	newTypeWriter(buf, qf).signature(sig)
}

// instanceMarker is the prefix for an instantiated type in unexpanded form.
const instanceMarker = '#'

type typeWriter struct {
	buf  *bytes.Buffer
	seen map[Type]bool
	qf   Qualifier
	hash bool
}

func newTypeWriter(buf *bytes.Buffer, qf Qualifier) *typeWriter {
	return &typeWriter{buf, make(map[Type]bool), qf, false}
}

func newTypeHasher(buf *bytes.Buffer) *typeWriter {
	return &typeWriter{buf, make(map[Type]bool), nil, true}
}

func (w *typeWriter) byte(b byte)                               { w.buf.WriteByte(b) }
func (w *typeWriter) string(s string)                           { w.buf.WriteString(s) }
func (w *typeWriter) writef(format string, args ...interface{}) { fmt.Fprintf(w.buf, format, args...) }
func (w *typeWriter) error(msg string) {
	if w.hash {
		panic(msg)
	}
	w.string("<" + msg + ">")
}

func (w *typeWriter) typ(typ Type) {
	if w.seen[typ] {
		w.error("cycle to " + goTypeName(typ))
		return
	}
	w.seen[typ] = true
	defer delete(w.seen, typ)

	switch t := typ.(type) {
	case nil:
		w.error("nil")

	case *Basic:
		// exported basic types go into package unsafe
		// (currently this is just unsafe.Pointer)
		if token.IsExported(t.name) {
			if obj, _ := Unsafe.scope.Lookup(t.name).(*TypeName); obj != nil {
				w.typeName(obj)
				break
			}
		}
		w.string(t.name)

	case *Array:
		w.writef("[%d]", t.len)
		w.typ(t.elem)

	case *Slice:
		w.string("[]")
		w.typ(t.elem)

	case *Struct:
		w.string("struct{")
		for i, f := range t.fields {
			if i > 0 {
				w.string("; ")
			}
			// This doesn't do the right thing for embedded type
			// aliases where we should print the alias name, not
			// the aliased type (see issue #44410).
			if !f.embedded {
				w.string(f.name)
				w.byte(' ')
			}
			w.typ(f.typ)
			if tag := t.Tag(i); tag != "" {
				w.writef(" %q", tag)
			}
		}
		w.byte('}')

	case *Pointer:
		w.byte('*')
		w.typ(t.base)

	case *Tuple:
		w.tuple(t, false)

	case *Signature:
		w.string("func")
		w.signature(t)

	case *Union:
		// Unions only appear as (syntactic) embedded elements
		// in interfaces and syntactically cannot be empty.
		if t.Len() == 0 {
			w.error("empty union")
			break
		}
		for i, t := range t.terms {
			if i > 0 {
				w.byte('|')
			}
			if t.tilde {
				w.byte('~')
			}
			w.typ(t.typ)
		}

	case *Interface:
		w.string("interface{")
		first := true
		for _, m := range t.methods {
			if !first {
				w.string("; ")
			}
			first = false
			w.string(m.name)
			w.signature(m.typ.(*Signature))
		}
		for _, typ := range t.embeddeds {
			if !first {
				w.string("; ")
			}
			first = false
			w.typ(typ)
		}
		w.byte('}')

	case *Map:
		w.string("map[")
		w.typ(t.key)
		w.byte(']')
		w.typ(t.elem)

	case *Chan:
		var s string
		var parens bool
		switch t.dir {
		case SendRecv:
			s = "chan "
			// chan (<-chan T) requires parentheses
			if c, _ := t.elem.(*Chan); c != nil && c.dir == RecvOnly {
				parens = true
			}
		case SendOnly:
			s = "chan<- "
		case RecvOnly:
			s = "<-chan "
		default:
			w.error("unknown channel direction")
		}
		w.string(s)
		if parens {
			w.byte('(')
		}
		w.typ(t.elem)
		if parens {
			w.byte(')')
		}

	case *Named:
		// Instance markers indicate unexpanded instantiated
		// types. Write them to aid debugging, but don't write
		// them when we need an instance hash: whether a type
		// is fully expanded or not doesn't matter for identity.
		if !w.hash && t.instPos != nil {
			w.byte(instanceMarker)
		}
		w.typeName(t.obj)
		if t.targs != nil {
			// instantiated type
			w.typeList(t.targs.list())
		} else if t.TParams().Len() != 0 {
			// parameterized type
			w.tParamList(t.TParams().list())
		}

	case *TypeParam:
		if t.obj == nil {
			w.error("unnamed type parameter")
			break
		}
		// Optionally write out package for typeparams (like Named).
		// TODO(danscales): this is required for import/export, so
		// we maybe need a separate function that won't be changed
		// for debugging purposes.
		if t.obj.pkg != nil {
			writePackage(w.buf, t.obj.pkg, w.qf)
		}
		w.string(t.obj.name + subscript(t.id))

	case *top:
		w.error("⊤")

	default:
		// For externally defined implementations of Type.
		// Note: In this case cycles won't be caught.
		w.string(t.String())
	}
}

func (w *typeWriter) typeList(list []Type) {
	w.byte('[')
	for i, typ := range list {
		if i > 0 {
			w.string(", ")
		}
		w.typ(typ)
	}
	w.byte(']')
}

func (w *typeWriter) tParamList(list []*TypeParam) {
	w.byte('[')
	var prev Type
	for i, tpar := range list {
		// Determine the type parameter and its constraint.
		// list is expected to hold type parameter names,
		// but don't crash if that's not the case.
		if tpar == nil {
			w.error("nil type parameter")
			continue
		}
		if i > 0 {
			if tpar.bound != prev {
				// bound changed - write previous one before advancing
				w.byte(' ')
				w.typ(prev)
			}
			w.string(", ")
		}
		prev = tpar.bound
		w.typ(tpar)
	}
	if prev != nil {
		w.byte(' ')
		w.typ(prev)
	}
	w.byte(']')
}

func (w *typeWriter) typeName(obj *TypeName) {
	if obj.pkg != nil {
		writePackage(w.buf, obj.pkg, w.qf)
	}
	w.string(obj.name)

	if w.hash {
		// For local defined types, use the (original!) TypeName's scope
		// numbers to disambiguate.
		if typ, _ := obj.typ.(*Named); typ != nil {
			// TODO(gri) Figure out why typ.orig != typ.orig.orig sometimes
			//           and whether the loop can iterate more than twice.
			//           (It seems somehow connected to instance types.)
			for typ.orig != typ {
				typ = typ.orig
			}
			w.writeScopeNumbers(typ.obj.parent)
		}
	}
}

// writeScopeNumbers writes the number sequence for this scope to buf
// in the form ".i.j.k" where i, j, k, etc. stand for scope numbers.
// If a scope is nil or has no parent (such as a package scope), nothing
// is written.
func (w *typeWriter) writeScopeNumbers(s *Scope) {
	if s != nil && s.number > 0 {
		w.writeScopeNumbers(s.parent)
		w.writef(".%d", s.number)
	}
}

func (w *typeWriter) tuple(tup *Tuple, variadic bool) {
	w.byte('(')
	if tup != nil {
		for i, v := range tup.vars {
			if i > 0 {
				w.string(", ")
			}
			// parameter names are ignored for type identity and thus type hashes
			if !w.hash && v.name != "" {
				w.string(v.name)
				w.byte(' ')
			}
			typ := v.typ
			if variadic && i == len(tup.vars)-1 {
				if s, ok := typ.(*Slice); ok {
					w.string("...")
					typ = s.elem
				} else {
					// special case:
					// append(s, "foo"...) leads to signature func([]byte, string...)
					if t := asBasic(typ); t == nil || t.kind != String {
						w.error("expected string type")
						continue
					}
					w.typ(typ)
					w.string("...")
					continue
				}
			}
			w.typ(typ)
		}
	}
	w.byte(')')
}

func (w *typeWriter) signature(sig *Signature) {
	if sig.TParams().Len() != 0 {
		w.tParamList(sig.TParams().list())
	}

	w.tuple(sig.params, sig.variadic)

	n := sig.results.Len()
	if n == 0 {
		// no result
		return
	}

	w.byte(' ')
	if n == 1 && (w.hash || sig.results.vars[0].name == "") {
		// single unnamed result (if type hashing, name must be ignored)
		w.typ(sig.results.vars[0].typ)
		return
	}

	// multiple or named result(s)
	w.tuple(sig.results, false)
}

// subscript returns the decimal (utf8) representation of x using subscript digits.
func subscript(x uint64) string {
	const w = len("₀") // all digits 0...9 have the same utf8 width
	var buf [32 * w]byte
	i := len(buf)
	for {
		i -= w
		utf8.EncodeRune(buf[i:], '₀'+rune(x%10)) // '₀' == U+2080
		x /= 10
		if x == 0 {
			break
		}
	}
	return string(buf[i:])
}
