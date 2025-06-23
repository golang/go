// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements Selections.

package types2

import (
	"bytes"
	"fmt"
)

// SelectionKind describes the kind of a selector expression x.f
// (excluding qualified identifiers).
//
// If x is a struct or *struct, a selector expression x.f may denote a
// sequence of selection operations x.a.b.c.f. The SelectionKind
// describes the kind of the final (explicit) operation; all the
// previous (implicit) operations are always field selections.
// Each element of Indices specifies an implicit field (a, b, c)
// by its index in the struct type of the field selection operand.
//
// For a FieldVal operation, the final selection refers to the field
// specified by Selection.Obj.
//
// For a MethodVal operation, the final selection refers to a method.
// If the "pointerness" of the method's declared receiver does not
// match that of the effective receiver after implicit field
// selection, then an & or * operation is implicitly applied to the
// receiver variable or value.
// So, x.f denotes (&x.a.b.c).f when f requires a pointer receiver but
// x.a.b.c is a non-pointer variable; and it denotes (*x.a.b.c).f when
// f requires a non-pointer receiver but x.a.b.c is a pointer value.
//
// All pointer indirections, whether due to implicit or explicit field
// selections or * operations inserted for "pointerness", panic if
// applied to a nil pointer, so a method call x.f() may panic even
// before the function call.
//
// By contrast, a MethodExpr operation T.f is essentially equivalent
// to a function literal of the form:
//
//	func(x T, args) (results) { return x.f(args) }
//
// Consequently, any implicit field selections and * operations
// inserted for "pointerness" are not evaluated until the function is
// called, so a T.f or (*T).f expression never panics.
type SelectionKind int

const (
	FieldVal   SelectionKind = iota // x.f is a struct field selector
	MethodVal                       // x.f is a method selector
	MethodExpr                      // x.f is a method expression
)

// A Selection describes a selector expression x.f.
// For the declarations:
//
//	type T struct{ x int; E }
//	type E struct{}
//	func (e E) m() {}
//	var p *T
//
// the following relations exist:
//
//	Selector    Kind          Recv    Obj    Type       Index     Indirect
//
//	p.x         FieldVal      T       x      int        {0}       true
//	p.m         MethodVal     *T      m      func()     {1, 0}    true
//	T.m         MethodExpr    T       m      func(T)    {1, 0}    false
type Selection struct {
	kind     SelectionKind
	recv     Type   // type of x
	obj      Object // object denoted by x.f
	index    []int  // path from x to x.f
	indirect bool   // set if there was any pointer indirection on the path
}

// Kind returns the selection kind.
func (s *Selection) Kind() SelectionKind { return s.kind }

// Recv returns the type of x in x.f.
func (s *Selection) Recv() Type { return s.recv }

// Obj returns the object denoted by x.f; a *Var for
// a field selection, and a *Func in all other cases.
func (s *Selection) Obj() Object { return s.obj }

// Type returns the type of x.f, which may be different from the type of f.
// See Selection for more information.
func (s *Selection) Type() Type {
	switch s.kind {
	case MethodVal:
		// The type of x.f is a method with its receiver type set
		// to the type of x.
		sig := *s.obj.(*Func).typ.(*Signature)
		recv := *sig.recv
		recv.typ = s.recv
		sig.recv = &recv
		return &sig

	case MethodExpr:
		// The type of x.f is a function (without receiver)
		// and an additional first argument with the same type as x.
		// TODO(gri) Similar code is already in call.go - factor!
		// TODO(gri) Compute this eagerly to avoid allocations.
		sig := *s.obj.(*Func).typ.(*Signature)
		arg0 := *sig.recv
		sig.recv = nil
		arg0.typ = s.recv
		var params []*Var
		if sig.params != nil {
			params = sig.params.vars
		}
		sig.params = NewTuple(append([]*Var{&arg0}, params...)...)
		return &sig
	}

	// In all other cases, the type of x.f is the type of x.
	return s.obj.Type()
}

// Index describes the path from x to f in x.f.
// The last index entry is the field or method index of the type declaring f;
// either:
//
//  1. the list of declared methods of a named type; or
//  2. the list of methods of an interface type; or
//  3. the list of fields of a struct type.
//
// The earlier index entries are the indices of the embedded fields implicitly
// traversed to get from (the type of) x to f, starting at embedding depth 0.
func (s *Selection) Index() []int { return s.index }

// Indirect reports whether any pointer indirection was required to get from
// x to f in x.f.
//
// Beware: Indirect spuriously returns true (Go issue #8353) for a
// MethodVal selection in which the receiver argument and parameter
// both have type *T so there is no indirection.
// Unfortunately, a fix is too risky.
func (s *Selection) Indirect() bool { return s.indirect }

func (s *Selection) String() string { return SelectionString(s, nil) }

// SelectionString returns the string form of s.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
//
// Examples:
//
//	"field (T) f int"
//	"method (T) f(X) Y"
//	"method expr (T) f(X) Y"
func SelectionString(s *Selection, qf Qualifier) string {
	var k string
	switch s.kind {
	case FieldVal:
		k = "field "
	case MethodVal:
		k = "method "
	case MethodExpr:
		k = "method expr "
	default:
		panic("unreachable")
	}
	var buf bytes.Buffer
	buf.WriteString(k)
	buf.WriteByte('(')
	WriteType(&buf, s.Recv(), qf)
	fmt.Fprintf(&buf, ") %s", s.obj.Name())
	if T := s.Type(); s.kind == FieldVal {
		buf.WriteByte(' ')
		WriteType(&buf, T, qf)
	} else {
		WriteSignature(&buf, T.(*Signature), qf)
	}
	return buf.String()
}
