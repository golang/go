// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package demangle

import (
	"bytes"
	"fmt"
	"strings"
)

// AST is an abstract syntax tree representing a C++ declaration.
// This is sufficient for the demangler but is by no means a general C++ AST.
type AST interface {
	// Internal method to convert to demangled string.
	print(*printState)

	// Traverse each element of an AST.  If the function returns
	// false, traversal of children of that element is skipped.
	Traverse(func(AST) bool)

	// Copy an AST with possible transformations.
	// If the skip function returns true, no copy is required.
	// If the copy function returns nil, no copy is required.
	// Otherwise the AST returned by copy is used in a copy of the full AST.
	// Copy itself returns either a copy or nil.
	Copy(copy func(AST) AST, skip func(AST) bool) AST

	// Implement the fmt.GoStringer interface.
	GoString() string
	goString(indent int, field string) string
}

// ASTToString returns the demangled name of the AST.
func ASTToString(a AST, options ...Option) string {
	tparams := true
	for _, o := range options {
		switch o {
		case NoTemplateParams:
			tparams = false
		}
	}

	ps := printState{tparams: tparams}
	a.print(&ps)
	return ps.buf.String()
}

// The printState type holds information needed to print an AST.
type printState struct {
	tparams bool // whether to print template parameters

	buf  bytes.Buffer
	last byte // Last byte written to buffer.

	// The inner field is a list of items to print for a type
	// name.  This is used by types to implement the inside-out
	// C++ declaration syntax.
	inner []AST

	// The printing field is a list of items we are currently
	// printing.  This avoids endless recursion if a substitution
	// reference creates a cycle in the graph.
	printing []AST
}

// writeByte adds a byte to the string being printed.
func (ps *printState) writeByte(b byte) {
	ps.last = b
	ps.buf.WriteByte(b)
}

// writeString adds a string to the string being printed.
func (ps *printState) writeString(s string) {
	if len(s) > 0 {
		ps.last = s[len(s)-1]
	}
	ps.buf.WriteString(s)
}

// Print an AST.
func (ps *printState) print(a AST) {
	c := 0
	for _, v := range ps.printing {
		if v == a {
			// We permit the type to appear once, and
			// return without printing anything if we see
			// it twice.  This is for a case like
			// _Z6outer2IsEPFilES1_, where the
			// substitution is printed differently the
			// second time because the set of inner types
			// is different.
			c++
			if c > 1 {
				return
			}
		}
	}
	ps.printing = append(ps.printing, a)

	a.print(ps)

	ps.printing = ps.printing[:len(ps.printing)-1]
}

// Name is an unqualified name.
type Name struct {
	Name string
}

func (n *Name) print(ps *printState) {
	ps.writeString(n.Name)
}

func (n *Name) Traverse(fn func(AST) bool) {
	fn(n)
}

func (n *Name) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(n) {
		return nil
	}
	return fn(n)
}

func (n *Name) GoString() string {
	return n.goString(0, "Name: ")
}

func (n *Name) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%s%s", indent, "", field, n.Name)
}

// Typed is a typed name.
type Typed struct {
	Name AST
	Type AST
}

func (t *Typed) print(ps *printState) {
	// We are printing a typed name, so ignore the current set of
	// inner names to print.  Pass down our name as the one to use.
	holdInner := ps.inner
	defer func() { ps.inner = holdInner }()

	ps.inner = []AST{t}
	ps.print(t.Type)
	if len(ps.inner) > 0 {
		// The type did not print the name; print it now in
		// the default location.
		ps.writeByte(' ')
		ps.print(t.Name)
	}
}

func (t *Typed) printInner(ps *printState) {
	ps.print(t.Name)
}

func (t *Typed) Traverse(fn func(AST) bool) {
	if fn(t) {
		t.Name.Traverse(fn)
		t.Type.Traverse(fn)
	}
}

func (t *Typed) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(t) {
		return nil
	}
	name := t.Name.Copy(fn, skip)
	typ := t.Type.Copy(fn, skip)
	if name == nil && typ == nil {
		return fn(t)
	}
	if name == nil {
		name = t.Name
	}
	if typ == nil {
		typ = t.Type
	}
	t = &Typed{Name: name, Type: typ}
	if r := fn(t); r != nil {
		return r
	}
	return t
}

func (t *Typed) GoString() string {
	return t.goString(0, "")
}

func (t *Typed) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sTyped:\n%s\n%s", indent, "", field,
		t.Name.goString(indent+2, "Name: "),
		t.Type.goString(indent+2, "Type: "))
}

// Qualified is a name in a scope.
type Qualified struct {
	Scope AST
	Name  AST

	// The LocalName field is true if this is parsed as a
	// <local-name>.  We shouldn't really need this, but in some
	// cases (for the unary sizeof operator) the standard
	// demangler prints a local name slightly differently.  We
	// keep track of this for compatibility.
	LocalName bool // A full local name encoding
}

func (q *Qualified) print(ps *printState) {
	ps.print(q.Scope)
	ps.writeString("::")
	ps.print(q.Name)
}

func (q *Qualified) Traverse(fn func(AST) bool) {
	if fn(q) {
		q.Scope.Traverse(fn)
		q.Name.Traverse(fn)
	}
}

func (q *Qualified) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(q) {
		return nil
	}
	scope := q.Scope.Copy(fn, skip)
	name := q.Name.Copy(fn, skip)
	if scope == nil && name == nil {
		return fn(q)
	}
	if scope == nil {
		scope = q.Scope
	}
	if name == nil {
		name = q.Name
	}
	q = &Qualified{Scope: scope, Name: name, LocalName: q.LocalName}
	if r := fn(q); r != nil {
		return r
	}
	return q
}

func (q *Qualified) GoString() string {
	return q.goString(0, "")
}

func (q *Qualified) goString(indent int, field string) string {
	s := ""
	if q.LocalName {
		s = " LocalName: true"
	}
	return fmt.Sprintf("%*s%sQualified:%s\n%s\n%s", indent, "", field,
		s, q.Scope.goString(indent+2, "Scope: "),
		q.Name.goString(indent+2, "Name: "))
}

// Template is a template with arguments.
type Template struct {
	Name AST
	Args []AST
}

func (t *Template) print(ps *printState) {
	// Inner types apply to the template as a whole, they don't
	// cross over into the template.
	holdInner := ps.inner
	defer func() { ps.inner = holdInner }()

	ps.inner = nil
	ps.print(t.Name)

	if !ps.tparams {
		// Do not print template parameters.
		return
	}
	// We need an extra space after operator<.
	if ps.last == '<' {
		ps.writeByte(' ')
	}

	ps.writeByte('<')
	first := true
	for _, a := range t.Args {
		if ps.isEmpty(a) {
			continue
		}
		if !first {
			ps.writeString(", ")
		}
		ps.print(a)
		first = false
	}
	if ps.last == '>' {
		// Avoid syntactic ambiguity in old versions of C++.
		ps.writeByte(' ')
	}
	ps.writeByte('>')
}

func (t *Template) Traverse(fn func(AST) bool) {
	if fn(t) {
		t.Name.Traverse(fn)
		for _, a := range t.Args {
			a.Traverse(fn)
		}
	}
}

func (t *Template) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(t) {
		return nil
	}
	name := t.Name.Copy(fn, skip)
	changed := name != nil
	args := make([]AST, len(t.Args))
	for i, a := range t.Args {
		ac := a.Copy(fn, skip)
		if ac == nil {
			args[i] = a
		} else {
			args[i] = ac
			changed = true
		}
	}
	if !changed {
		return fn(t)
	}
	if name == nil {
		name = t.Name
	}
	t = &Template{Name: name, Args: args}
	if r := fn(t); r != nil {
		return r
	}
	return t
}

func (t *Template) GoString() string {
	return t.goString(0, "")
}

func (t *Template) goString(indent int, field string) string {
	var args string
	if len(t.Args) == 0 {
		args = fmt.Sprintf("%*sArgs: nil", indent+2, "")
	} else {
		args = fmt.Sprintf("%*sArgs:", indent+2, "")
		for i, a := range t.Args {
			args += "\n"
			args += a.goString(indent+4, fmt.Sprintf("%d: ", i))
		}
	}
	return fmt.Sprintf("%*s%sTemplate (%p):\n%s\n%s", indent, "", field, t,
		t.Name.goString(indent+2, "Name: "), args)
}

// TemplateParam is a template parameter.  The Template field is
// filled in while parsing the demangled string.  We don't normally
// see these while printing--they are replaced by the simplify
// function.
type TemplateParam struct {
	Index    int
	Template *Template
}

func (tp *TemplateParam) print(ps *printState) {
	if tp.Template == nil {
		panic("TemplateParam Template field is nil")
	}
	if tp.Index >= len(tp.Template.Args) {
		panic("TemplateParam Index out of bounds")
	}
	ps.print(tp.Template.Args[tp.Index])
}

func (tp *TemplateParam) Traverse(fn func(AST) bool) {
	fn(tp)
	// Don't traverse Template--it points elsewhere in the AST.
}

func (tp *TemplateParam) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(tp) {
		return nil
	}
	return fn(tp)
}

func (tp *TemplateParam) GoString() string {
	return tp.goString(0, "")
}

func (tp *TemplateParam) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sTemplateParam: Template: %p; Index %d", indent, "", field, tp.Template, tp.Index)
}

// Qualifiers is an ordered list of type qualifiers.
type Qualifiers []string

// TypeWithQualifiers is a type with standard qualifiers.
type TypeWithQualifiers struct {
	Base       AST
	Qualifiers Qualifiers
}

func (twq *TypeWithQualifiers) print(ps *printState) {
	// Give the base type a chance to print the inner types.
	ps.inner = append(ps.inner, twq)
	ps.print(twq.Base)
	if len(ps.inner) > 0 {
		// The qualifier wasn't printed by Base.
		ps.writeByte(' ')
		ps.writeString(strings.Join(twq.Qualifiers, " "))
		ps.inner = ps.inner[:len(ps.inner)-1]
	}
}

// Print qualifiers as an inner type by just printing the qualifiers.
func (twq *TypeWithQualifiers) printInner(ps *printState) {
	ps.writeByte(' ')
	ps.writeString(strings.Join(twq.Qualifiers, " "))
}

func (twq *TypeWithQualifiers) Traverse(fn func(AST) bool) {
	if fn(twq) {
		twq.Base.Traverse(fn)
	}
}

func (twq *TypeWithQualifiers) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(twq) {
		return nil
	}
	base := twq.Base.Copy(fn, skip)
	if base == nil {
		return fn(twq)
	}
	twq = &TypeWithQualifiers{Base: base, Qualifiers: twq.Qualifiers}
	if r := fn(twq); r != nil {
		return r
	}
	return twq
}

func (twq *TypeWithQualifiers) GoString() string {
	return twq.goString(0, "")
}

func (twq *TypeWithQualifiers) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sTypeWithQualifiers: Qualifiers: %s\n%s", indent, "", field,
		twq.Qualifiers, twq.Base.goString(indent+2, "Base: "))
}

// MethodWithQualifiers is a method with qualifiers.
type MethodWithQualifiers struct {
	Method       AST
	Qualifiers   Qualifiers
	RefQualifier string // "" or "&" or "&&"
}

func (mwq *MethodWithQualifiers) print(ps *printState) {
	// Give the base type a chance to print the inner types.
	ps.inner = append(ps.inner, mwq)
	ps.print(mwq.Method)
	if len(ps.inner) > 0 {
		if len(mwq.Qualifiers) > 0 {
			ps.writeByte(' ')
			ps.writeString(strings.Join(mwq.Qualifiers, " "))
		}
		if mwq.RefQualifier != "" {
			ps.writeByte(' ')
			ps.writeString(mwq.RefQualifier)
		}
		ps.inner = ps.inner[:len(ps.inner)-1]
	}
}

func (mwq *MethodWithQualifiers) printInner(ps *printState) {
	if len(mwq.Qualifiers) > 0 {
		ps.writeByte(' ')
		ps.writeString(strings.Join(mwq.Qualifiers, " "))
	}
	if mwq.RefQualifier != "" {
		ps.writeByte(' ')
		ps.writeString(mwq.RefQualifier)
	}
}

func (mwq *MethodWithQualifiers) Traverse(fn func(AST) bool) {
	if fn(mwq) {
		mwq.Method.Traverse(fn)
	}
}

func (mwq *MethodWithQualifiers) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(mwq) {
		return nil
	}
	method := mwq.Method.Copy(fn, skip)
	if method == nil {
		return fn(mwq)
	}
	mwq = &MethodWithQualifiers{Method: method, Qualifiers: mwq.Qualifiers, RefQualifier: mwq.RefQualifier}
	if r := fn(mwq); r != nil {
		return r
	}
	return mwq
}

func (mwq *MethodWithQualifiers) GoString() string {
	return mwq.goString(0, "")
}

func (mwq *MethodWithQualifiers) goString(indent int, field string) string {
	var q string
	if len(mwq.Qualifiers) > 0 {
		q += fmt.Sprintf(" Qualifiers: %v", mwq.Qualifiers)
	}
	if mwq.RefQualifier != "" {
		if q != "" {
			q += ";"
		}
		q += " RefQualifier: " + mwq.RefQualifier
	}
	return fmt.Sprintf("%*s%sMethodWithQualifiers:%s\n%s", indent, "", field,
		q, mwq.Method.goString(indent+2, "Method: "))
}

// BuiltinType is a builtin type, like "int".
type BuiltinType struct {
	Name string
}

func (bt *BuiltinType) print(ps *printState) {
	ps.writeString(bt.Name)
}

func (bt *BuiltinType) Traverse(fn func(AST) bool) {
	fn(bt)
}

func (bt *BuiltinType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(bt) {
		return nil
	}
	return fn(bt)
}

func (bt *BuiltinType) GoString() string {
	return bt.goString(0, "")
}

func (bt *BuiltinType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sBuiltinType: %s", indent, "", field, bt.Name)
}

// printBase is common print code for types that are printed with a
// simple suffix.
func printBase(ps *printState, qual, base AST) {
	ps.inner = append(ps.inner, qual)
	ps.print(base)
	if len(ps.inner) > 0 {
		qual.(innerPrinter).printInner(ps)
		ps.inner = ps.inner[:len(ps.inner)-1]
	}
}

// PointerType is a pointer type.
type PointerType struct {
	Base AST
}

func (pt *PointerType) print(ps *printState) {
	printBase(ps, pt, pt.Base)
}

func (pt *PointerType) printInner(ps *printState) {
	ps.writeString("*")
}

func (pt *PointerType) Traverse(fn func(AST) bool) {
	if fn(pt) {
		pt.Base.Traverse(fn)
	}
}

func (pt *PointerType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(pt) {
		return nil
	}
	base := pt.Base.Copy(fn, skip)
	if base == nil {
		return fn(pt)
	}
	pt = &PointerType{Base: base}
	if r := fn(pt); r != nil {
		return r
	}
	return pt
}

func (pt *PointerType) GoString() string {
	return pt.goString(0, "")
}

func (pt *PointerType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sPointerType:\n%s", indent, "", field,
		pt.Base.goString(indent+2, ""))
}

// ReferenceType is a reference type.
type ReferenceType struct {
	Base AST
}

func (rt *ReferenceType) print(ps *printState) {
	printBase(ps, rt, rt.Base)
}

func (rt *ReferenceType) printInner(ps *printState) {
	ps.writeString("&")
}

func (rt *ReferenceType) Traverse(fn func(AST) bool) {
	if fn(rt) {
		rt.Base.Traverse(fn)
	}
}

func (rt *ReferenceType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(rt) {
		return nil
	}
	base := rt.Base.Copy(fn, skip)
	if base == nil {
		return fn(rt)
	}
	rt = &ReferenceType{Base: base}
	if r := fn(rt); r != nil {
		return r
	}
	return rt
}

func (rt *ReferenceType) GoString() string {
	return rt.goString(0, "")
}

func (rt *ReferenceType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sReferenceType:\n%s", indent, "", field,
		rt.Base.goString(indent+2, ""))
}

// RvalueReferenceType is an rvalue reference type.
type RvalueReferenceType struct {
	Base AST
}

func (rt *RvalueReferenceType) print(ps *printState) {
	printBase(ps, rt, rt.Base)
}

func (rt *RvalueReferenceType) printInner(ps *printState) {
	ps.writeString("&&")
}

func (rt *RvalueReferenceType) Traverse(fn func(AST) bool) {
	if fn(rt) {
		rt.Base.Traverse(fn)
	}
}

func (rt *RvalueReferenceType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(rt) {
		return nil
	}
	base := rt.Base.Copy(fn, skip)
	if base == nil {
		return fn(rt)
	}
	rt = &RvalueReferenceType{Base: base}
	if r := fn(rt); r != nil {
		return r
	}
	return rt
}

func (rt *RvalueReferenceType) GoString() string {
	return rt.goString(0, "")
}

func (rt *RvalueReferenceType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sRvalueReferenceType:\n%s", indent, "", field,
		rt.Base.goString(indent+2, ""))
}

// ComplexType is a complex type.
type ComplexType struct {
	Base AST
}

func (ct *ComplexType) print(ps *printState) {
	printBase(ps, ct, ct.Base)
}

func (ct *ComplexType) printInner(ps *printState) {
	ps.writeString(" _Complex")
}

func (ct *ComplexType) Traverse(fn func(AST) bool) {
	if fn(ct) {
		ct.Base.Traverse(fn)
	}
}

func (ct *ComplexType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(ct) {
		return nil
	}
	base := ct.Base.Copy(fn, skip)
	if base == nil {
		return fn(ct)
	}
	ct = &ComplexType{Base: base}
	if r := fn(ct); r != nil {
		return r
	}
	return ct
}

func (ct *ComplexType) GoString() string {
	return ct.goString(0, "")
}

func (ct *ComplexType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sComplexType:\n%s", indent, "", field,
		ct.Base.goString(indent+2, ""))
}

// ImaginaryType is an imaginary type.
type ImaginaryType struct {
	Base AST
}

func (it *ImaginaryType) print(ps *printState) {
	printBase(ps, it, it.Base)
}

func (it *ImaginaryType) printInner(ps *printState) {
	ps.writeString(" _Imaginary")
}

func (it *ImaginaryType) Traverse(fn func(AST) bool) {
	if fn(it) {
		it.Base.Traverse(fn)
	}
}

func (it *ImaginaryType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(it) {
		return nil
	}
	base := it.Base.Copy(fn, skip)
	if base == nil {
		return fn(it)
	}
	it = &ImaginaryType{Base: base}
	if r := fn(it); r != nil {
		return r
	}
	return it
}

func (it *ImaginaryType) GoString() string {
	return it.goString(0, "")
}

func (it *ImaginaryType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sImaginaryType:\n%s", indent, "", field,
		it.Base.goString(indent+2, ""))
}

// VendorQualifier is a type qualified by a vendor-specific qualifier.
type VendorQualifier struct {
	Qualifier AST
	Type      AST
}

func (vq *VendorQualifier) print(ps *printState) {
	ps.inner = append(ps.inner, vq)
	ps.print(vq.Type)
	if len(ps.inner) > 0 {
		ps.printOneInner(nil)
	}
}

func (vq *VendorQualifier) printInner(ps *printState) {
	ps.writeByte(' ')
	ps.print(vq.Qualifier)
}

func (vq *VendorQualifier) Traverse(fn func(AST) bool) {
	if fn(vq) {
		vq.Qualifier.Traverse(fn)
		vq.Type.Traverse(fn)
	}
}

func (vq *VendorQualifier) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(vq) {
		return nil
	}
	qualifier := vq.Qualifier.Copy(fn, skip)
	typ := vq.Type.Copy(fn, skip)
	if qualifier == nil && typ == nil {
		return fn(vq)
	}
	if qualifier == nil {
		qualifier = vq.Qualifier
	}
	if typ == nil {
		typ = vq.Type
	}
	vq = &VendorQualifier{Qualifier: qualifier, Type: vq.Type}
	if r := fn(vq); r != nil {
		return r
	}
	return vq
}

func (vq *VendorQualifier) GoString() string {
	return vq.goString(0, "")
}

func (vq *VendorQualifier) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sVendorQualifier:\n%s\n%s", indent, "", field,
		vq.Qualifier.goString(indent+2, "Qualifier: "),
		vq.Type.goString(indent+2, "Type: "))
}

// ArrayType is an array type.
type ArrayType struct {
	Dimension AST
	Element   AST
}

func (at *ArrayType) print(ps *printState) {
	// Pass the array type down as an inner type so that we print
	// multi-dimensional arrays correctly.
	ps.inner = append(ps.inner, at)
	ps.print(at.Element)
	if ln := len(ps.inner); ln > 0 {
		ps.inner = ps.inner[:ln-1]
		at.printDimension(ps)
	}
}

func (at *ArrayType) printInner(ps *printState) {
	at.printDimension(ps)
}

// Print the array dimension.
func (at *ArrayType) printDimension(ps *printState) {
	space := " "
	for len(ps.inner) > 0 {
		// We haven't gotten to the real type yet.  Use
		// parentheses around that type, except that if it is
		// an array type we print it as a multi-dimensional
		// array
		in := ps.inner[len(ps.inner)-1]
		if twq, ok := in.(*TypeWithQualifiers); ok {
			in = twq.Base
		}
		if _, ok := in.(*ArrayType); ok {
			if in == ps.inner[len(ps.inner)-1] {
				space = ""
			}
			ps.printOneInner(nil)
		} else {
			ps.writeString(" (")
			ps.printInner(false)
			ps.writeByte(')')
		}
	}
	ps.writeString(space)
	ps.writeByte('[')
	ps.print(at.Dimension)
	ps.writeByte(']')
}

func (at *ArrayType) Traverse(fn func(AST) bool) {
	if fn(at) {
		at.Dimension.Traverse(fn)
		at.Element.Traverse(fn)
	}
}

func (at *ArrayType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(at) {
		return nil
	}
	dimension := at.Dimension.Copy(fn, skip)
	element := at.Element.Copy(fn, skip)
	if dimension == nil && element == nil {
		return fn(at)
	}
	if dimension == nil {
		dimension = at.Dimension
	}
	if element == nil {
		element = at.Element
	}
	at = &ArrayType{Dimension: dimension, Element: element}
	if r := fn(at); r != nil {
		return r
	}
	return at
}

func (at *ArrayType) GoString() string {
	return at.goString(0, "")
}

func (at *ArrayType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sArrayType:\n%s\n%s", indent, "", field,
		at.Dimension.goString(indent+2, "Dimension: "),
		at.Element.goString(indent+2, "Element: "))
}

// FunctionType is a function type.  The Return field may be nil for
// cases where the return type is not part of the mangled name.
type FunctionType struct {
	Return AST
	Args   []AST
}

func (ft *FunctionType) print(ps *printState) {
	if ft.Return != nil {
		// Pass the return type as an inner type in order to
		// print the arguments in the right location.
		ps.inner = append(ps.inner, ft)
		ps.print(ft.Return)
		if len(ps.inner) == 0 {
			// Everything was printed.
			return
		}
		ps.inner = ps.inner[:len(ps.inner)-1]
		ps.writeByte(' ')
	}
	ft.printArgs(ps)
}

func (ft *FunctionType) printInner(ps *printState) {
	ft.printArgs(ps)
}

// printArgs prints the arguments of a function type.  It looks at the
// inner types for spacing.
func (ft *FunctionType) printArgs(ps *printState) {
	paren := false
	space := false
	for i := len(ps.inner) - 1; i >= 0; i-- {
		switch ps.inner[i].(type) {
		case *PointerType, *ReferenceType, *RvalueReferenceType:
			paren = true
		case *TypeWithQualifiers, *ComplexType, *ImaginaryType, *PtrMem:
			space = true
			paren = true
		}
		if paren {
			break
		}
	}

	if paren {
		if !space && (ps.last != '(' && ps.last != '*') {
			space = true
		}
		if space && ps.last != ' ' {
			ps.writeByte(' ')
		}
		ps.writeByte('(')
	}

	save := ps.printInner(true)

	if paren {
		ps.writeByte(')')
	}

	ps.writeByte('(')
	first := true
	for _, a := range ft.Args {
		if ps.isEmpty(a) {
			continue
		}
		if !first {
			ps.writeString(", ")
		}
		ps.print(a)
		first = false
	}
	ps.writeByte(')')

	ps.inner = save
	ps.printInner(false)
}

func (ft *FunctionType) Traverse(fn func(AST) bool) {
	if fn(ft) {
		if ft.Return != nil {
			ft.Return.Traverse(fn)
		}
		for _, a := range ft.Args {
			a.Traverse(fn)
		}
	}
}

func (ft *FunctionType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(ft) {
		return nil
	}
	changed := false
	var ret AST
	if ft.Return != nil {
		ret = ft.Return.Copy(fn, skip)
		if ret == nil {
			ret = ft.Return
		} else {
			changed = true
		}
	}
	args := make([]AST, len(ft.Args))
	for i, a := range ft.Args {
		ac := a.Copy(fn, skip)
		if ac == nil {
			args[i] = a
		} else {
			args[i] = ac
			changed = true
		}
	}
	if !changed {
		return fn(ft)
	}
	ft = &FunctionType{Return: ret, Args: args}
	if r := fn(ft); r != nil {
		return r
	}
	return ft
}

func (ft *FunctionType) GoString() string {
	return ft.goString(0, "")
}

func (ft *FunctionType) goString(indent int, field string) string {
	var r string
	if ft.Return == nil {
		r = fmt.Sprintf("%*sReturn: nil", indent+2, "")
	} else {
		r = ft.Return.goString(indent+2, "Return: ")
	}
	var args string
	if len(ft.Args) == 0 {
		args = fmt.Sprintf("%*sArgs: nil", indent+2, "")
	} else {
		args = fmt.Sprintf("%*sArgs:", indent+2, "")
		for i, a := range ft.Args {
			args += "\n"
			args += a.goString(indent+4, fmt.Sprintf("%d: ", i))
		}
	}
	return fmt.Sprintf("%*s%sFunctionType:\n%s\n%s", indent, "", field, r, args)
}

// FunctionParam is a parameter of a function, used for last-specified
// return type in a closure.
type FunctionParam struct {
	Index int
}

func (fp *FunctionParam) print(ps *printState) {
	if fp.Index == 0 {
		ps.writeString("this")
	} else {
		fmt.Fprintf(&ps.buf, "{parm#%d}", fp.Index)
	}
}

func (fp *FunctionParam) Traverse(fn func(AST) bool) {
	fn(fp)
}

func (fp *FunctionParam) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(fp) {
		return nil
	}
	return fn(fp)
}

func (fp *FunctionParam) GoString() string {
	return fp.goString(0, "")
}

func (fp *FunctionParam) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sFunctionParam: %d", indent, "", field, fp.Index)
}

// PtrMem is a pointer-to-member expression.
type PtrMem struct {
	Class  AST
	Member AST
}

func (pm *PtrMem) print(ps *printState) {
	ps.inner = append(ps.inner, pm)
	ps.print(pm.Member)
	if len(ps.inner) > 0 {
		ps.printOneInner(nil)
	}
}

func (pm *PtrMem) printInner(ps *printState) {
	if ps.last != '(' {
		ps.writeByte(' ')
	}
	ps.print(pm.Class)
	ps.writeString("::*")
}

func (pm *PtrMem) Traverse(fn func(AST) bool) {
	if fn(pm) {
		pm.Class.Traverse(fn)
		pm.Member.Traverse(fn)
	}
}

func (pm *PtrMem) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(pm) {
		return nil
	}
	class := pm.Class.Copy(fn, skip)
	member := pm.Member.Copy(fn, skip)
	if class == nil && member == nil {
		return fn(pm)
	}
	if class == nil {
		class = pm.Class
	}
	if member == nil {
		member = pm.Member
	}
	pm = &PtrMem{Class: class, Member: member}
	if r := fn(pm); r != nil {
		return r
	}
	return pm
}

func (pm *PtrMem) GoString() string {
	return pm.goString(0, "")
}

func (pm *PtrMem) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sPtrMem:\n%s\n%s", indent, "", field,
		pm.Class.goString(indent+2, "Class: "),
		pm.Member.goString(indent+2, "Member: "))
}

// FixedType is a fixed numeric type of unknown size.
type FixedType struct {
	Base  AST
	Accum bool
	Sat   bool
}

func (ft *FixedType) print(ps *printState) {
	if ft.Sat {
		ps.writeString("_Sat ")
	}
	if bt, ok := ft.Base.(*BuiltinType); ok && bt.Name == "int" {
		// The standard demangler skips printing "int".
	} else {
		ps.print(ft.Base)
		ps.writeByte(' ')
	}
	if ft.Accum {
		ps.writeString("_Accum")
	} else {
		ps.writeString("_Fract")
	}
}

func (ft *FixedType) Traverse(fn func(AST) bool) {
	if fn(ft) {
		ft.Base.Traverse(fn)
	}
}

func (ft *FixedType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(ft) {
		return nil
	}
	base := ft.Base.Copy(fn, skip)
	if base == nil {
		return fn(ft)
	}
	ft = &FixedType{Base: base, Accum: ft.Accum, Sat: ft.Sat}
	if r := fn(ft); r != nil {
		return r
	}
	return ft
}

func (ft *FixedType) GoString() string {
	return ft.goString(0, "")
}

func (ft *FixedType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sFixedType: Accum: %t; Sat: %t\n%s", indent, "", field,
		ft.Accum, ft.Sat,
		ft.Base.goString(indent+2, "Base: "))
}

// VectorType is a vector type.
type VectorType struct {
	Dimension AST
	Base      AST
}

func (vt *VectorType) print(ps *printState) {
	ps.inner = append(ps.inner, vt)
	ps.print(vt.Base)
	if len(ps.inner) > 0 {
		ps.printOneInner(nil)
	}
}

func (vt *VectorType) printInner(ps *printState) {
	ps.writeString(" __vector(")
	ps.print(vt.Dimension)
	ps.writeByte(')')
}

func (vt *VectorType) Traverse(fn func(AST) bool) {
	if fn(vt) {
		vt.Dimension.Traverse(fn)
		vt.Base.Traverse(fn)
	}
}

func (vt *VectorType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(vt) {
		return nil
	}
	dimension := vt.Dimension.Copy(fn, skip)
	base := vt.Base.Copy(fn, skip)
	if dimension == nil && base == nil {
		return fn(vt)
	}
	if dimension == nil {
		dimension = vt.Dimension
	}
	if base == nil {
		base = vt.Base
	}
	vt = &VectorType{Dimension: dimension, Base: base}
	if r := fn(vt); r != nil {
		return r
	}
	return vt
}

func (vt *VectorType) GoString() string {
	return vt.goString(0, "")
}

func (vt *VectorType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sVectorType:\n%s\n%s", indent, "", field,
		vt.Dimension.goString(indent+2, "Dimension: "),
		vt.Base.goString(indent+2, "Base: "))
}

// Decltype is the decltype operator.
type Decltype struct {
	Expr AST
}

func (dt *Decltype) print(ps *printState) {
	ps.writeString("decltype (")
	ps.print(dt.Expr)
	ps.writeByte(')')
}

func (dt *Decltype) Traverse(fn func(AST) bool) {
	if fn(dt) {
		dt.Expr.Traverse(fn)
	}
}

func (dt *Decltype) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(dt) {
		return nil
	}
	expr := dt.Expr.Copy(fn, skip)
	if expr == nil {
		return fn(dt)
	}
	dt = &Decltype{Expr: expr}
	if r := fn(dt); r != nil {
		return r
	}
	return dt
}

func (dt *Decltype) GoString() string {
	return dt.goString(0, "")
}

func (dt *Decltype) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sDecltype:\n%s", indent, "", field,
		dt.Expr.goString(indent+2, "Expr: "))
}

// Operator is an operator.
type Operator struct {
	Name string
}

func (op *Operator) print(ps *printState) {
	ps.writeString("operator")
	if isLower(op.Name[0]) {
		ps.writeByte(' ')
	}
	n := op.Name
	n = strings.TrimSuffix(n, " ")
	ps.writeString(n)
}

func (op *Operator) Traverse(fn func(AST) bool) {
	fn(op)
}

func (op *Operator) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(op) {
		return nil
	}
	return fn(op)
}

func (op *Operator) GoString() string {
	return op.goString(0, "")
}

func (op *Operator) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sOperator: %s", indent, "", field, op.Name)
}

// Constructor is a constructor.
type Constructor struct {
	Name AST
}

func (c *Constructor) print(ps *printState) {
	ps.print(c.Name)
}

func (c *Constructor) Traverse(fn func(AST) bool) {
	if fn(c) {
		c.Name.Traverse(fn)
	}
}

func (c *Constructor) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(c) {
		return nil
	}
	name := c.Name.Copy(fn, skip)
	if name == nil {
		return fn(c)
	}
	c = &Constructor{Name: name}
	if r := fn(c); r != nil {
		return r
	}
	return c
}

func (c *Constructor) GoString() string {
	return c.goString(0, "")
}

func (c *Constructor) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sConstructor:\n%s", indent, "", field, c.Name.goString(indent+2, "Name: "))
}

// Destructor is a destructor.
type Destructor struct {
	Name AST
}

func (d *Destructor) print(ps *printState) {
	ps.writeByte('~')
	ps.print(d.Name)
}

func (d *Destructor) Traverse(fn func(AST) bool) {
	if fn(d) {
		d.Name.Traverse(fn)
	}
}

func (d *Destructor) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(d) {
		return nil
	}
	name := d.Name.Copy(fn, skip)
	if name == nil {
		return fn(d)
	}
	d = &Destructor{Name: name}
	if r := fn(d); r != nil {
		return r
	}
	return d
}

func (d *Destructor) GoString() string {
	return d.goString(0, "")
}

func (d *Destructor) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sDestructor:\n%s", indent, "", field, d.Name.goString(indent+2, "Name: "))
}

// GlobalCDtor is a global constructor or destructor.
type GlobalCDtor struct {
	Ctor bool
	Key  AST
}

func (gcd *GlobalCDtor) print(ps *printState) {
	ps.writeString("global ")
	if gcd.Ctor {
		ps.writeString("constructors")
	} else {
		ps.writeString("destructors")
	}
	ps.writeString(" keyed to ")
	ps.print(gcd.Key)
}

func (gcd *GlobalCDtor) Traverse(fn func(AST) bool) {
	if fn(gcd) {
		gcd.Key.Traverse(fn)
	}
}

func (gcd *GlobalCDtor) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(gcd) {
		return nil
	}
	key := gcd.Key.Copy(fn, skip)
	if key == nil {
		return fn(gcd)
	}
	gcd = &GlobalCDtor{Ctor: gcd.Ctor, Key: key}
	if r := fn(gcd); r != nil {
		return r
	}
	return gcd
}

func (gcd *GlobalCDtor) GoString() string {
	return gcd.goString(0, "")
}

func (gcd *GlobalCDtor) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sGlobalCDtor: Ctor: %t\n%s", indent, "", field,
		gcd.Ctor, gcd.Key.goString(indent+2, "Key: "))
}

// TaggedName is a name with an ABI tag.
type TaggedName struct {
	Name AST
	Tag  AST
}

func (t *TaggedName) print(ps *printState) {
	ps.print(t.Name)
	ps.writeString("[abi:")
	ps.print(t.Tag)
	ps.writeByte(']')
}

func (t *TaggedName) Traverse(fn func(AST) bool) {
	if fn(t) {
		t.Name.Traverse(fn)
		t.Tag.Traverse(fn)
	}
}

func (t *TaggedName) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(t) {
		return nil
	}
	name := t.Name.Copy(fn, skip)
	tag := t.Tag.Copy(fn, skip)
	if name == nil && tag == nil {
		return fn(t)
	}
	if name == nil {
		name = t.Name
	}
	if tag == nil {
		tag = t.Tag
	}
	t = &TaggedName{Name: name, Tag: tag}
	if r := fn(t); r != nil {
		return r
	}
	return t
}

func (t *TaggedName) GoString() string {
	return t.goString(0, "")
}

func (t *TaggedName) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sTaggedName:\n%s\n%s", indent, "", field,
		t.Name.goString(indent+2, "Name: "),
		t.Tag.goString(indent+2, "Tag: "))
}

// PackExpansion is a pack expansion.  The Pack field may be nil.
type PackExpansion struct {
	Base AST
	Pack *ArgumentPack
}

func (pe *PackExpansion) print(ps *printState) {
	// We normally only get here if the simplify function was
	// unable to locate and expand the pack.
	if pe.Pack == nil {
		parenthesize(ps, pe.Base)
		ps.writeString("...")
	} else {
		ps.print(pe.Base)
	}
}

func (pe *PackExpansion) Traverse(fn func(AST) bool) {
	if fn(pe) {
		pe.Base.Traverse(fn)
		// Don't traverse Template--it points elsewhere in the AST.
	}
}

func (pe *PackExpansion) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(pe) {
		return nil
	}
	base := pe.Base.Copy(fn, skip)
	if base == nil {
		return fn(pe)
	}
	pe = &PackExpansion{Base: base, Pack: pe.Pack}
	if r := fn(pe); r != nil {
		return r
	}
	return pe
}

func (pe *PackExpansion) GoString() string {
	return pe.goString(0, "")
}

func (pe *PackExpansion) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sPackExpansion: Pack: %p\n%s", indent, "", field,
		pe.Pack, pe.Base.goString(indent+2, "Base: "))
}

// ArgumentPack is an argument pack.
type ArgumentPack struct {
	Args []AST
}

func (ap *ArgumentPack) print(ps *printState) {
	for i, a := range ap.Args {
		if i > 0 {
			ps.writeString(", ")
		}
		ps.print(a)
	}
}

func (ap *ArgumentPack) Traverse(fn func(AST) bool) {
	if fn(ap) {
		for _, a := range ap.Args {
			a.Traverse(fn)
		}
	}
}

func (ap *ArgumentPack) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(ap) {
		return nil
	}
	args := make([]AST, len(ap.Args))
	changed := false
	for i, a := range ap.Args {
		ac := a.Copy(fn, skip)
		if ac == nil {
			args[i] = a
		} else {
			args[i] = ac
			changed = true
		}
	}
	if !changed {
		return fn(ap)
	}
	ap = &ArgumentPack{Args: args}
	if r := fn(ap); r != nil {
		return r
	}
	return ap
}

func (ap *ArgumentPack) GoString() string {
	return ap.goString(0, "")
}

func (ap *ArgumentPack) goString(indent int, field string) string {
	if len(ap.Args) == 0 {
		return fmt.Sprintf("%*s%sArgumentPack: nil", indent, "", field)
	}
	s := fmt.Sprintf("%*s%sArgumentPack:", indent, "", field)
	for i, a := range ap.Args {
		s += "\n"
		s += a.goString(indent+2, fmt.Sprintf("%d: ", i))
	}
	return s
}

// SizeofPack is the sizeof operator applied to an argument pack.
type SizeofPack struct {
	Pack *ArgumentPack
}

func (sp *SizeofPack) print(ps *printState) {
	ps.writeString(fmt.Sprintf("%d", len(sp.Pack.Args)))
}

func (sp *SizeofPack) Traverse(fn func(AST) bool) {
	fn(sp)
	// Don't traverse the pack--it points elsewhere in the AST.
}

func (sp *SizeofPack) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(sp) {
		return nil
	}
	sp = &SizeofPack{Pack: sp.Pack}
	if r := fn(sp); r != nil {
		return r
	}
	return sp
}

func (sp *SizeofPack) GoString() string {
	return sp.goString(0, "")
}

func (sp *SizeofPack) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sSizeofPack: Pack: %p", indent, "", field, sp.Pack)
}

// SizeofArgs is the size of a captured template parameter pack from
// an alias template.
type SizeofArgs struct {
	Args []AST
}

func (sa *SizeofArgs) print(ps *printState) {
	c := 0
	for _, a := range sa.Args {
		if ap, ok := a.(*ArgumentPack); ok {
			c += len(ap.Args)
		} else if el, ok := a.(*ExprList); ok {
			c += len(el.Exprs)
		} else {
			c++
		}
	}
	ps.writeString(fmt.Sprintf("%d", c))
}

func (sa *SizeofArgs) Traverse(fn func(AST) bool) {
	if fn(sa) {
		for _, a := range sa.Args {
			a.Traverse(fn)
		}
	}
}

func (sa *SizeofArgs) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(sa) {
		return nil
	}
	changed := false
	args := make([]AST, len(sa.Args))
	for i, a := range sa.Args {
		ac := a.Copy(fn, skip)
		if ac == nil {
			args[i] = a
		} else {
			args[i] = ac
			changed = true
		}
	}
	if !changed {
		return fn(sa)
	}
	sa = &SizeofArgs{Args: args}
	if r := fn(sa); r != nil {
		return r
	}
	return sa
}

func (sa *SizeofArgs) GoString() string {
	return sa.goString(0, "")
}

func (sa *SizeofArgs) goString(indent int, field string) string {
	var args string
	if len(sa.Args) == 0 {
		args = fmt.Sprintf("%*sArgs: nil", indent+2, "")
	} else {
		args = fmt.Sprintf("%*sArgs:", indent+2, "")
		for i, a := range sa.Args {
			args += "\n"
			args += a.goString(indent+4, fmt.Sprintf("%d: ", i))
		}
	}
	return fmt.Sprintf("%*s%sSizeofArgs:\n%s", indent, "", field, args)
}

// Cast is a type cast.
type Cast struct {
	To AST
}

func (c *Cast) print(ps *printState) {
	ps.writeString("operator ")
	ps.print(c.To)
}

func (c *Cast) Traverse(fn func(AST) bool) {
	if fn(c) {
		c.To.Traverse(fn)
	}
}

func (c *Cast) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(c) {
		return nil
	}
	to := c.To.Copy(fn, skip)
	if to == nil {
		return fn(c)
	}
	c = &Cast{To: to}
	if r := fn(c); r != nil {
		return r
	}
	return c
}

func (c *Cast) GoString() string {
	return c.goString(0, "")
}

func (c *Cast) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sCast\n%s", indent, "", field,
		c.To.goString(indent+2, "To: "))
}

// The parenthesize function prints the string for val, wrapped in
// parentheses if necessary.
func parenthesize(ps *printState, val AST) {
	paren := false
	switch v := val.(type) {
	case *Name, *InitializerList, *FunctionParam:
	case *Qualified:
		if v.LocalName {
			paren = true
		}
	default:
		paren = true
	}
	if paren {
		ps.writeByte('(')
	}
	ps.print(val)
	if paren {
		ps.writeByte(')')
	}
}

// Nullary is an operator in an expression with no arguments, such as
// throw.
type Nullary struct {
	Op AST
}

func (n *Nullary) print(ps *printState) {
	if op, ok := n.Op.(*Operator); ok {
		ps.writeString(op.Name)
	} else {
		ps.print(n.Op)
	}
}

func (n *Nullary) Traverse(fn func(AST) bool) {
	if fn(n) {
		n.Op.Traverse(fn)
	}
}

func (n *Nullary) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(n) {
		return nil
	}
	op := n.Op.Copy(fn, skip)
	if op == nil {
		return fn(n)
	}
	n = &Nullary{Op: op}
	if r := fn(n); r != nil {
		return r
	}
	return n
}

func (n *Nullary) GoString() string {
	return n.goString(0, "")
}

func (n *Nullary) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sNullary:\n%s", indent, "", field,
		n.Op.goString(indent+2, "Op: "))
}

// Unary is a unary operation in an expression.
type Unary struct {
	Op         AST
	Expr       AST
	Suffix     bool // true for ++ -- when used as postfix
	SizeofType bool // true for sizeof (type)
}

func (u *Unary) print(ps *printState) {
	expr := u.Expr

	// Don't print the argument list when taking the address of a
	// function.
	if op, ok := u.Op.(*Operator); ok && op.Name == "&" {
		if t, ok := expr.(*Typed); ok {
			if _, ok := t.Type.(*FunctionType); ok {
				expr = t.Name
			}
		}
	}

	if u.Suffix {
		parenthesize(ps, expr)
	}

	if op, ok := u.Op.(*Operator); ok {
		ps.writeString(op.Name)
	} else if c, ok := u.Op.(*Cast); ok {
		ps.writeByte('(')
		ps.print(c.To)
		ps.writeByte(')')
	} else {
		ps.print(u.Op)
	}

	if !u.Suffix {
		if op, ok := u.Op.(*Operator); ok && op.Name == "::" {
			// Don't use parentheses after ::.
			ps.print(expr)
		} else if u.SizeofType {
			// Always use parentheses for sizeof argument.
			ps.writeByte('(')
			ps.print(expr)
			ps.writeByte(')')
		} else {
			parenthesize(ps, expr)
		}
	}
}

func (u *Unary) Traverse(fn func(AST) bool) {
	if fn(u) {
		u.Op.Traverse(fn)
		u.Expr.Traverse(fn)
	}
}

func (u *Unary) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(u) {
		return nil
	}
	op := u.Op.Copy(fn, skip)
	expr := u.Expr.Copy(fn, skip)
	if op == nil && expr == nil {
		return fn(u)
	}
	if op == nil {
		op = u.Op
	}
	if expr == nil {
		expr = u.Expr
	}
	u = &Unary{Op: op, Expr: expr, Suffix: u.Suffix, SizeofType: u.SizeofType}
	if r := fn(u); r != nil {
		return r
	}
	return u
}

func (u *Unary) GoString() string {
	return u.goString(0, "")
}

func (u *Unary) goString(indent int, field string) string {
	var s string
	if u.Suffix {
		s = " Suffix: true"
	}
	if u.SizeofType {
		s += " SizeofType: true"
	}
	return fmt.Sprintf("%*s%sUnary:%s\n%s\n%s", indent, "", field,
		s, u.Op.goString(indent+2, "Op: "),
		u.Expr.goString(indent+2, "Expr: "))
}

// Binary is a binary operation in an expression.
type Binary struct {
	Op    AST
	Left  AST
	Right AST
}

func (b *Binary) print(ps *printState) {
	op, _ := b.Op.(*Operator)

	if op != nil && strings.Contains(op.Name, "cast") {
		ps.writeString(op.Name)
		ps.writeByte('<')
		ps.print(b.Left)
		ps.writeString(">(")
		ps.print(b.Right)
		ps.writeByte(')')
		return
	}

	// Use an extra set of parentheses around an expression that
	// uses the greater-than operator, so that it does not get
	// confused with the '>' that ends template parameters.
	if op != nil && op.Name == ">" {
		ps.writeByte('(')
	}

	left := b.Left

	// A function call in an expression should not print the types
	// of the arguments.
	if op != nil && op.Name == "()" {
		if ty, ok := b.Left.(*Typed); ok {
			left = ty.Name
		}
	}

	parenthesize(ps, left)

	if op != nil && op.Name == "[]" {
		ps.writeByte('[')
		ps.print(b.Right)
		ps.writeByte(']')
		return
	}

	if op != nil {
		if op.Name != "()" {
			ps.writeString(op.Name)
		}
	} else {
		ps.print(b.Op)
	}

	parenthesize(ps, b.Right)

	if op != nil && op.Name == ">" {
		ps.writeByte(')')
	}
}

func (b *Binary) Traverse(fn func(AST) bool) {
	if fn(b) {
		b.Op.Traverse(fn)
		b.Left.Traverse(fn)
		b.Right.Traverse(fn)
	}
}

func (b *Binary) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(b) {
		return nil
	}
	op := b.Op.Copy(fn, skip)
	left := b.Left.Copy(fn, skip)
	right := b.Right.Copy(fn, skip)
	if op == nil && left == nil && right == nil {
		return fn(b)
	}
	if op == nil {
		op = b.Op
	}
	if left == nil {
		left = b.Left
	}
	if right == nil {
		right = b.Right
	}
	b = &Binary{Op: op, Left: left, Right: right}
	if r := fn(b); r != nil {
		return r
	}
	return b
}

func (b *Binary) GoString() string {
	return b.goString(0, "")
}

func (b *Binary) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sBinary:\n%s\n%s\n%s", indent, "", field,
		b.Op.goString(indent+2, "Op: "),
		b.Left.goString(indent+2, "Left: "),
		b.Right.goString(indent+2, "Right: "))
}

// Trinary is the ?: trinary operation in an expression.
type Trinary struct {
	Op     AST
	First  AST
	Second AST
	Third  AST
}

func (t *Trinary) print(ps *printState) {
	parenthesize(ps, t.First)
	ps.writeByte('?')
	parenthesize(ps, t.Second)
	ps.writeString(" : ")
	parenthesize(ps, t.Third)
}

func (t *Trinary) Traverse(fn func(AST) bool) {
	if fn(t) {
		t.Op.Traverse(fn)
		t.First.Traverse(fn)
		t.Second.Traverse(fn)
		t.Third.Traverse(fn)
	}
}

func (t *Trinary) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(t) {
		return nil
	}
	op := t.Op.Copy(fn, skip)
	first := t.First.Copy(fn, skip)
	second := t.Second.Copy(fn, skip)
	third := t.Third.Copy(fn, skip)
	if op == nil && first == nil && second == nil && third == nil {
		return fn(t)
	}
	if op == nil {
		op = t.Op
	}
	if first == nil {
		first = t.First
	}
	if second == nil {
		second = t.Second
	}
	if third == nil {
		third = t.Third
	}
	t = &Trinary{Op: op, First: first, Second: second, Third: third}
	if r := fn(t); r != nil {
		return r
	}
	return t
}

func (t *Trinary) GoString() string {
	return t.goString(0, "")
}

func (t *Trinary) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sTrinary:\n%s\n%s\n%s\n%s", indent, "", field,
		t.Op.goString(indent+2, "Op: "),
		t.First.goString(indent+2, "First: "),
		t.Second.goString(indent+2, "Second: "),
		t.Third.goString(indent+2, "Third: "))
}

// Fold is a C++17 fold-expression.  Arg2 is nil for a unary operator.
type Fold struct {
	Left bool
	Op   AST
	Arg1 AST
	Arg2 AST
}

func (f *Fold) print(ps *printState) {
	op, _ := f.Op.(*Operator)
	printOp := func() {
		if op != nil {
			ps.writeString(op.Name)
		} else {
			ps.print(f.Op)
		}
	}

	if f.Arg2 == nil {
		if f.Left {
			ps.writeString("(...")
			printOp()
			parenthesize(ps, f.Arg1)
			ps.writeString(")")
		} else {
			ps.writeString("(")
			parenthesize(ps, f.Arg1)
			printOp()
			ps.writeString("...)")
		}
	} else {
		ps.writeString("(")
		parenthesize(ps, f.Arg1)
		printOp()
		ps.writeString("...")
		printOp()
		parenthesize(ps, f.Arg2)
		ps.writeString(")")
	}
}

func (f *Fold) Traverse(fn func(AST) bool) {
	if fn(f) {
		f.Op.Traverse(fn)
		f.Arg1.Traverse(fn)
		if f.Arg2 != nil {
			f.Arg2.Traverse(fn)
		}
	}
}

func (f *Fold) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(f) {
		return nil
	}
	op := f.Op.Copy(fn, skip)
	arg1 := f.Arg1.Copy(fn, skip)
	var arg2 AST
	if f.Arg2 != nil {
		arg2 = f.Arg2.Copy(fn, skip)
	}
	if op == nil && arg1 == nil && arg2 == nil {
		return fn(f)
	}
	if op == nil {
		op = f.Op
	}
	if arg1 == nil {
		arg1 = f.Arg1
	}
	if arg2 == nil {
		arg2 = f.Arg2
	}
	f = &Fold{Left: f.Left, Op: op, Arg1: arg1, Arg2: arg2}
	if r := fn(f); r != nil {
		return r
	}
	return f
}

func (f *Fold) GoString() string {
	return f.goString(0, "")
}

func (f *Fold) goString(indent int, field string) string {
	if f.Arg2 == nil {
		return fmt.Sprintf("%*s%sFold: Left: %t\n%s\n%s", indent, "", field,
			f.Left, f.Op.goString(indent+2, "Op: "),
			f.Arg1.goString(indent+2, "Arg1: "))
	} else {
		return fmt.Sprintf("%*s%sFold: Left: %t\n%s\n%s\n%s", indent, "", field,
			f.Left, f.Op.goString(indent+2, "Op: "),
			f.Arg1.goString(indent+2, "Arg1: "),
			f.Arg2.goString(indent+2, "Arg2: "))
	}
}

// New is a use of operator new in an expression.
type New struct {
	Op    AST
	Place AST
	Type  AST
	Init  AST
}

func (n *New) print(ps *printState) {
	// Op doesn't really matter for printing--we always print "new".
	ps.writeString("new ")
	if n.Place != nil {
		parenthesize(ps, n.Place)
		ps.writeByte(' ')
	}
	ps.print(n.Type)
	if n.Init != nil {
		parenthesize(ps, n.Init)
	}
}

func (n *New) Traverse(fn func(AST) bool) {
	if fn(n) {
		n.Op.Traverse(fn)
		if n.Place != nil {
			n.Place.Traverse(fn)
		}
		n.Type.Traverse(fn)
		if n.Init != nil {
			n.Init.Traverse(fn)
		}
	}
}

func (n *New) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(n) {
		return nil
	}
	op := n.Op.Copy(fn, skip)
	var place AST
	if n.Place != nil {
		place = n.Place.Copy(fn, skip)
	}
	typ := n.Type.Copy(fn, skip)
	var ini AST
	if n.Init != nil {
		ini = n.Init.Copy(fn, skip)
	}
	if op == nil && place == nil && typ == nil && ini == nil {
		return fn(n)
	}
	if op == nil {
		op = n.Op
	}
	if place == nil {
		place = n.Place
	}
	if typ == nil {
		typ = n.Type
	}
	if ini == nil {
		ini = n.Init
	}
	n = &New{Op: op, Place: place, Type: typ, Init: ini}
	if r := fn(n); r != nil {
		return r
	}
	return n
}

func (n *New) GoString() string {
	return n.goString(0, "")
}

func (n *New) goString(indent int, field string) string {
	var place string
	if n.Place == nil {
		place = fmt.Sprintf("%*sPlace: nil", indent, "")
	} else {
		place = n.Place.goString(indent+2, "Place: ")
	}
	var ini string
	if n.Init == nil {
		ini = fmt.Sprintf("%*sInit: nil", indent, "")
	} else {
		ini = n.Init.goString(indent+2, "Init: ")
	}
	return fmt.Sprintf("%*s%sNew:\n%s\n%s\n%s\n%s", indent, "", field,
		n.Op.goString(indent+2, "Op: "), place,
		n.Type.goString(indent+2, "Type: "), ini)
}

// Literal is a literal in an expression.
type Literal struct {
	Type AST
	Val  string
	Neg  bool
}

// Suffixes to use for constants of the given integer type.
var builtinTypeSuffix = map[string]string{
	"int":                "",
	"unsigned int":       "u",
	"long":               "l",
	"unsigned long":      "ul",
	"long long":          "ll",
	"unsigned long long": "ull",
}

// Builtin float types.
var builtinTypeFloat = map[string]bool{
	"double":      true,
	"long double": true,
	"float":       true,
	"__float128":  true,
	"half":        true,
}

func (l *Literal) print(ps *printState) {
	isFloat := false
	if b, ok := l.Type.(*BuiltinType); ok {
		if suffix, ok := builtinTypeSuffix[b.Name]; ok {
			if l.Neg {
				ps.writeByte('-')
			}
			ps.writeString(l.Val)
			ps.writeString(suffix)
			return
		} else if b.Name == "bool" && !l.Neg {
			switch l.Val {
			case "0":
				ps.writeString("false")
				return
			case "1":
				ps.writeString("true")
				return
			}
		} else {
			isFloat = builtinTypeFloat[b.Name]
		}
	}

	ps.writeByte('(')
	ps.print(l.Type)
	ps.writeByte(')')

	if isFloat {
		ps.writeByte('[')
	}
	if l.Neg {
		ps.writeByte('-')
	}
	ps.writeString(l.Val)
	if isFloat {
		ps.writeByte(']')
	}
}

func (l *Literal) Traverse(fn func(AST) bool) {
	if fn(l) {
		l.Type.Traverse(fn)
	}
}

func (l *Literal) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(l) {
		return nil
	}
	typ := l.Type.Copy(fn, skip)
	if typ == nil {
		return fn(l)
	}
	l = &Literal{Type: typ, Val: l.Val, Neg: l.Neg}
	if r := fn(l); r != nil {
		return r
	}
	return l
}

func (l *Literal) GoString() string {
	return l.goString(0, "")
}

func (l *Literal) goString(indent int, field string) string {
	var neg string
	if l.Neg {
		neg = " Neg: true"
	}
	return fmt.Sprintf("%*s%sLiteral:%s\n%s\n%*sVal: %s", indent, "", field,
		neg, l.Type.goString(indent+2, "Type: "),
		indent+2, "", l.Val)
}

// ExprList is a list of expressions, typically arguments to a
// function call in an expression.
type ExprList struct {
	Exprs []AST
}

func (el *ExprList) print(ps *printState) {
	for i, e := range el.Exprs {
		if i > 0 {
			ps.writeString(", ")
		}
		ps.print(e)
	}
}

func (el *ExprList) Traverse(fn func(AST) bool) {
	if fn(el) {
		for _, e := range el.Exprs {
			e.Traverse(fn)
		}
	}
}

func (el *ExprList) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(el) {
		return nil
	}
	exprs := make([]AST, len(el.Exprs))
	changed := false
	for i, e := range el.Exprs {
		ec := e.Copy(fn, skip)
		if ec == nil {
			exprs[i] = e
		} else {
			exprs[i] = ec
			changed = true
		}
	}
	if !changed {
		return fn(el)
	}
	el = &ExprList{Exprs: exprs}
	if r := fn(el); r != nil {
		return r
	}
	return el
}

func (el *ExprList) GoString() string {
	return el.goString(0, "")
}

func (el *ExprList) goString(indent int, field string) string {
	if len(el.Exprs) == 0 {
		return fmt.Sprintf("%*s%sExprList: nil", indent, "", field)
	}
	s := fmt.Sprintf("%*s%sExprList:", indent, "", field)
	for i, e := range el.Exprs {
		s += "\n"
		s += e.goString(indent+2, fmt.Sprintf("%d: ", i))
	}
	return s
}

// InitializerList is an initializer list: an optional type with a
// list of expressions.
type InitializerList struct {
	Type  AST
	Exprs AST
}

func (il *InitializerList) print(ps *printState) {
	if il.Type != nil {
		ps.print(il.Type)
	}
	ps.writeByte('{')
	ps.print(il.Exprs)
	ps.writeByte('}')
}

func (il *InitializerList) Traverse(fn func(AST) bool) {
	if fn(il) {
		if il.Type != nil {
			il.Type.Traverse(fn)
		}
		il.Exprs.Traverse(fn)
	}
}

func (il *InitializerList) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(il) {
		return nil
	}
	var typ AST
	if il.Type != nil {
		typ = il.Type.Copy(fn, skip)
	}
	exprs := il.Exprs.Copy(fn, skip)
	if typ == nil && exprs == nil {
		return fn(il)
	}
	if typ == nil {
		typ = il.Type
	}
	if exprs == nil {
		exprs = il.Exprs
	}
	il = &InitializerList{Type: typ, Exprs: exprs}
	if r := fn(il); r != nil {
		return r
	}
	return il
}

func (il *InitializerList) GoString() string {
	return il.goString(0, "")
}

func (il *InitializerList) goString(indent int, field string) string {
	var t string
	if il.Type == nil {
		t = fmt.Sprintf("%*sType: nil", indent+2, "")
	} else {
		t = il.Type.goString(indent+2, "Type: ")
	}
	return fmt.Sprintf("%*s%sInitializerList:\n%s\n%s", indent, "", field,
		t, il.Exprs.goString(indent+2, "Exprs: "))
}

// DefaultArg holds a default argument for a local name.
type DefaultArg struct {
	Num int
	Arg AST
}

func (da *DefaultArg) print(ps *printState) {
	fmt.Fprintf(&ps.buf, "{default arg#%d}::", da.Num+1)
	ps.print(da.Arg)
}

func (da *DefaultArg) Traverse(fn func(AST) bool) {
	if fn(da) {
		da.Arg.Traverse(fn)
	}
}

func (da *DefaultArg) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(da) {
		return nil
	}
	arg := da.Arg.Copy(fn, skip)
	if arg == nil {
		return fn(da)
	}
	da = &DefaultArg{Num: da.Num, Arg: arg}
	if r := fn(da); r != nil {
		return r
	}
	return da
}

func (da *DefaultArg) GoString() string {
	return da.goString(0, "")
}

func (da *DefaultArg) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sDefaultArg: Num: %d\n%s", indent, "", field, da.Num,
		da.Arg.goString(indent+2, "Arg: "))
}

// Closure is a closure, or lambda expression.
type Closure struct {
	Types []AST
	Num   int
}

func (cl *Closure) print(ps *printState) {
	ps.writeString("{lambda(")
	for i, t := range cl.Types {
		if i > 0 {
			ps.writeString(", ")
		}
		ps.print(t)
	}
	ps.writeString(fmt.Sprintf(")#%d}", cl.Num+1))
}

func (cl *Closure) Traverse(fn func(AST) bool) {
	if fn(cl) {
		for _, t := range cl.Types {
			t.Traverse(fn)
		}
	}
}

func (cl *Closure) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(cl) {
		return nil
	}
	types := make([]AST, len(cl.Types))
	changed := false
	for i, t := range cl.Types {
		tc := t.Copy(fn, skip)
		if tc == nil {
			types[i] = t
		} else {
			types[i] = tc
			changed = true
		}
	}
	if !changed {
		return fn(cl)
	}
	cl = &Closure{Types: types, Num: cl.Num}
	if r := fn(cl); r != nil {
		return r
	}
	return cl
}

func (cl *Closure) GoString() string {
	return cl.goString(0, "")
}

func (cl *Closure) goString(indent int, field string) string {
	var types string
	if len(cl.Types) == 0 {
		types = fmt.Sprintf("%*sTypes: nil", indent+2, "")
	} else {
		types = fmt.Sprintf("%*sTypes:", indent+2, "")
		for i, t := range cl.Types {
			types += "\n"
			types += t.goString(indent+4, fmt.Sprintf("%d: ", i))
		}
	}
	return fmt.Sprintf("%*s%sClosure: Num: %d\n%s", indent, "", field, cl.Num, types)
}

// UnnamedType is an unnamed type, that just has an index.
type UnnamedType struct {
	Num int
}

func (ut *UnnamedType) print(ps *printState) {
	ps.writeString(fmt.Sprintf("{unnamed type#%d}", ut.Num+1))
}

func (ut *UnnamedType) Traverse(fn func(AST) bool) {
	fn(ut)
}

func (ut *UnnamedType) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(ut) {
		return nil
	}
	return fn(ut)
}

func (ut *UnnamedType) GoString() string {
	return ut.goString(0, "")
}

func (ut *UnnamedType) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sUnnamedType: Num: %d", indent, "", field, ut.Num)
}

// Clone is a clone of a function, with a distinguishing suffix.
type Clone struct {
	Base   AST
	Suffix string
}

func (c *Clone) print(ps *printState) {
	ps.print(c.Base)
	ps.writeString(fmt.Sprintf(" [clone %s]", c.Suffix))
}

func (c *Clone) Traverse(fn func(AST) bool) {
	if fn(c) {
		c.Base.Traverse(fn)
	}
}

func (c *Clone) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(c) {
		return nil
	}
	base := c.Base.Copy(fn, skip)
	if base == nil {
		return fn(c)
	}
	c = &Clone{Base: base, Suffix: c.Suffix}
	if r := fn(c); r != nil {
		return r
	}
	return c
}

func (c *Clone) GoString() string {
	return c.goString(0, "")
}

func (c *Clone) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sClone: Suffix: %s\n%s", indent, "", field,
		c.Suffix, c.Base.goString(indent+2, "Base: "))
}

// Special is a special symbol, printed as a prefix plus another
// value.
type Special struct {
	Prefix string
	Val    AST
}

func (s *Special) print(ps *printState) {
	ps.writeString(s.Prefix)
	ps.print(s.Val)
}

func (s *Special) Traverse(fn func(AST) bool) {
	if fn(s) {
		s.Val.Traverse(fn)
	}
}

func (s *Special) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(s) {
		return nil
	}
	val := s.Val.Copy(fn, skip)
	if val == nil {
		return fn(s)
	}
	s = &Special{Prefix: s.Prefix, Val: val}
	if r := fn(s); r != nil {
		return r
	}
	return s
}

func (s *Special) GoString() string {
	return s.goString(0, "")
}

func (s *Special) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sSpecial: Prefix: %s\n%s", indent, "", field,
		s.Prefix, s.Val.goString(indent+2, "Val: "))
}

// Special2 is like special, but uses two values.
type Special2 struct {
	Prefix string
	Val1   AST
	Middle string
	Val2   AST
}

func (s *Special2) print(ps *printState) {
	ps.writeString(s.Prefix)
	ps.print(s.Val1)
	ps.writeString(s.Middle)
	ps.print(s.Val2)
}

func (s *Special2) Traverse(fn func(AST) bool) {
	if fn(s) {
		s.Val1.Traverse(fn)
		s.Val2.Traverse(fn)
	}
}

func (s *Special2) Copy(fn func(AST) AST, skip func(AST) bool) AST {
	if skip(s) {
		return nil
	}
	val1 := s.Val1.Copy(fn, skip)
	val2 := s.Val2.Copy(fn, skip)
	if val1 == nil && val2 == nil {
		return fn(s)
	}
	if val1 == nil {
		val1 = s.Val1
	}
	if val2 == nil {
		val2 = s.Val2
	}
	s = &Special2{Prefix: s.Prefix, Val1: val1, Middle: s.Middle, Val2: val2}
	if r := fn(s); r != nil {
		return r
	}
	return s
}

func (s *Special2) GoString() string {
	return s.goString(0, "")
}

func (s *Special2) goString(indent int, field string) string {
	return fmt.Sprintf("%*s%sSpecial2: Prefix: %s\n%s\n%*sMiddle: %s\n%s", indent, "", field,
		s.Prefix, s.Val1.goString(indent+2, "Val1: "),
		indent+2, "", s.Middle, s.Val2.goString(indent+2, "Val2: "))
}

// Print the inner types.
func (ps *printState) printInner(prefixOnly bool) []AST {
	var save []AST
	var psave *[]AST
	if prefixOnly {
		psave = &save
	}
	for len(ps.inner) > 0 {
		ps.printOneInner(psave)
	}
	return save
}

// innerPrinter is an interface for types that can print themselves as
// inner types.
type innerPrinter interface {
	printInner(*printState)
}

// Print the most recent inner type.  If save is not nil, only print
// prefixes.
func (ps *printState) printOneInner(save *[]AST) {
	if len(ps.inner) == 0 {
		panic("printOneInner called with no inner types")
	}
	ln := len(ps.inner)
	a := ps.inner[ln-1]
	ps.inner = ps.inner[:ln-1]

	if save != nil {
		if _, ok := a.(*MethodWithQualifiers); ok {
			*save = append(*save, a)
			return
		}
	}

	if ip, ok := a.(innerPrinter); ok {
		ip.printInner(ps)
	} else {
		ps.print(a)
	}
}

// isEmpty returns whether printing a will not print anything.
func (ps *printState) isEmpty(a AST) bool {
	switch a := a.(type) {
	case *ArgumentPack:
		return len(a.Args) == 0
	case *ExprList:
		return len(a.Exprs) == 0
	case *PackExpansion:
		return a.Pack != nil && ps.isEmpty(a.Base)
	default:
		return false
	}
}
