// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"cmp"
	"fmt"
	"go/token"
	"io"
	"slices"
	"strings"
)

// Finding is an individual finding in a [Report].
type Finding interface {
	String() string
	Decl() Decl
}

// Report collects all findings from specfill across checked declarations.
// It implements the error interface.
type Report struct {
	TypeMismatches     []TypeMismatch
	NameMismatches     []NameMismatch
	DocOrderViolations []DocOrderViolation
	UnknownDecls       []UnknownDecl
	UnexpectedDocs     []UnexpectedDoc
}

// Error formats the report as a string, implementing the error interface.
func (r *Report) Error() string {
	var b strings.Builder
	r.Print(&b)
	return b.String()
}

// Decl identifies an AST declaration and its location in the source code.
type Decl struct {
	Pos  token.Position
	Recv string
	Name string
}

// Compare orders declarations by source position (filename, line, column,
// offset), breaking ties by receiver and name.
func (d Decl) Compare(other Decl) int {
	if c := cmp.Compare(d.Pos.Filename, other.Pos.Filename); c != 0 {
		return c
	}
	if c := cmp.Compare(d.Pos.Offset, other.Pos.Offset); c != 0 {
		return c
	}
	if c := cmp.Compare(d.Recv, other.Recv); c != 0 {
		return c
	}
	return cmp.Compare(d.Name, other.Name)
}

// String formats the declaration as "(Recv) Name" or "Name".
func (d Decl) String() string {
	if d.Recv != "" {
		return fmt.Sprintf("(%s) %s", d.Recv, d.Name)
	}
	return d.Name
}

// TypeMismatch records a disagreement between spec parameter/result types
// and an AST declaration's parameter/result types.
type TypeMismatch struct {
	D       Decl
	DeclSig string
	SpecSig string
	Details []string
}

func (m TypeMismatch) Decl() Decl { return m.D }

func (m TypeMismatch) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %s: type mismatch\n", m.D.Pos, m.D)
	fmt.Fprintf(&b, "  decl: %s\n", m.DeclSig)
	fmt.Fprintf(&b, "  spec: %s\n", m.SpecSig)
	for _, d := range m.Details {
		fmt.Fprintf(&b, "  - %s\n", d)
	}
	return b.String()
}

// NameMismatch records a disagreement between spec parameter/result names
// and an AST declaration's parameter/result names.
type NameMismatch struct {
	D       Decl
	DeclSig string
	SpecSig string
	Details []string
}

func (m NameMismatch) Decl() Decl { return m.D }

func (m NameMismatch) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %s: name mismatch\n", m.D.Pos, m.D)
	fmt.Fprintf(&b, "  decl: %s\n", m.DeclSig)
	fmt.Fprintf(&b, "  spec: %s\n", m.SpecSig)
	for _, d := range m.Details {
		fmt.Fprintf(&b, "  - %s\n", d)
	}
	return b.String()
}

// DocOrderViolation records a comment ordering violation in an exported
// declaration's doc comment, which would result in non-idempotent rewriting.
type DocOrderViolation struct {
	D   Decl
	Err error
}

func (v DocOrderViolation) Decl() Decl { return v.D }

func (v DocOrderViolation) String() string {
	return fmt.Sprintf("%s: %s: %v\n", v.D.Pos, v.D, v.Err)
}

// UnknownDecl records an exported declaration that has no matching entry in spec.
type UnknownDecl struct {
	D Decl
}

func (u UnknownDecl) Decl() Decl { return u.D }

func (u UnknownDecl) String() string {
	return fmt.Sprintf("%s: %s: declaration not in spec\n", u.D.Pos, u.D)
}

// UnexpectedDoc records an exported declaration containing an existing
// spec-owned doc comment when AllowDocRewrite is false.
type UnexpectedDoc struct {
	D Decl
}

func (u UnexpectedDoc) Decl() Decl { return u.D }

func (u UnexpectedDoc) String() string {
	return fmt.Sprintf("%s: %s: declaration contains unexpected doc comment\n", u.D.Pos, u.D)
}

// Merge combines other into r.
func (r *Report) Merge(other *Report) {
	r.TypeMismatches = append(r.TypeMismatches, other.TypeMismatches...)
	r.NameMismatches = append(r.NameMismatches, other.NameMismatches...)
	r.DocOrderViolations = append(r.DocOrderViolations, other.DocOrderViolations...)
	r.UnknownDecls = append(r.UnknownDecls, other.UnknownDecls...)
	r.UnexpectedDocs = append(r.UnexpectedDocs, other.UnexpectedDocs...)
}

// Empty reports whether the report contains no items.
func (r Report) Empty() bool {
	return len(r.TypeMismatches) == 0 &&
		len(r.NameMismatches) == 0 &&
		len(r.DocOrderViolations) == 0 &&
		len(r.UnknownDecls) == 0 &&
		len(r.UnexpectedDocs) == 0
}

func convertAppend[From Finding](ys []Finding, xs []From) []Finding {
	for _, x := range xs {
		ys = append(ys, x)
	}
	return ys
}

// Findings returns all findings in r as a slice of [Finding].
func (r Report) Findings() []Finding {
	var items []Finding
	items = convertAppend(items, r.TypeMismatches)
	items = convertAppend(items, r.NameMismatches)
	items = convertAppend(items, r.DocOrderViolations)
	items = convertAppend(items, r.UnknownDecls)
	items = convertAppend(items, r.UnexpectedDocs)
	return items
}

// Print writes the report to w sorted by position. If there are no findings,
// nothing is written.
func (r Report) Print(w io.Writer) {
	items := r.Findings()
	slices.SortStableFunc(items, func(a, b Finding) int {
		if c := a.Decl().Compare(b.Decl()); c != 0 {
			return c
		}
		return cmp.Compare(a.String(), b.String())
	})

	for _, item := range items {
		fmt.Fprint(w, item.String())
	}
}
