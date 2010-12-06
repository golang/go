// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import "go/token"

// ----------------------------------------------------------------------------
// Export filtering

func identListExports(list []*Ident) []*Ident {
	j := 0
	for _, x := range list {
		if x.IsExported() {
			list[j] = x
			j++
		}
	}
	return list[0:j]
}


// isExportedType assumes that typ is a correct type.
func isExportedType(typ Expr) bool {
	switch t := typ.(type) {
	case *Ident:
		return t.IsExported()
	case *ParenExpr:
		return isExportedType(t.X)
	case *SelectorExpr:
		// assume t.X is a typename
		return t.Sel.IsExported()
	case *StarExpr:
		return isExportedType(t.X)
	}
	return false
}


func fieldListExports(fields *FieldList, incomplete *bool) {
	if fields == nil {
		return
	}
	list := fields.List
	j := 0
	for _, f := range list {
		exported := false
		if len(f.Names) == 0 {
			// anonymous field
			// (Note that a non-exported anonymous field
			// may still refer to a type with exported
			// fields, so this is not absolutely correct.
			// However, this cannot be done w/o complete
			// type information.)
			exported = isExportedType(f.Type)
		} else {
			n := len(f.Names)
			f.Names = identListExports(f.Names)
			if len(f.Names) < n {
				*incomplete = true
			}
			exported = len(f.Names) > 0
		}
		if exported {
			typeExports(f.Type)
			list[j] = f
			j++
		}
	}
	if j < len(list) {
		*incomplete = true
	}
	fields.List = list[0:j]
}


func paramListExports(fields *FieldList) {
	if fields == nil {
		return
	}
	for _, f := range fields.List {
		typeExports(f.Type)
	}
}


func typeExports(typ Expr) {
	switch t := typ.(type) {
	case *ArrayType:
		typeExports(t.Elt)
	case *StructType:
		fieldListExports(t.Fields, &t.Incomplete)
	case *FuncType:
		paramListExports(t.Params)
		paramListExports(t.Results)
	case *InterfaceType:
		fieldListExports(t.Methods, &t.Incomplete)
	case *MapType:
		typeExports(t.Key)
		typeExports(t.Value)
	case *ChanType:
		typeExports(t.Value)
	}
}


func specExports(spec Spec) bool {
	switch s := spec.(type) {
	case *ValueSpec:
		s.Names = identListExports(s.Names)
		if len(s.Names) > 0 {
			typeExports(s.Type)
			return true
		}
	case *TypeSpec:
		if s.Name.IsExported() {
			typeExports(s.Type)
			return true
		}
	}
	return false
}


func specListExports(list []Spec) []Spec {
	j := 0
	for _, s := range list {
		if specExports(s) {
			list[j] = s
			j++
		}
	}
	return list[0:j]
}


func declExports(decl Decl) bool {
	switch d := decl.(type) {
	case *GenDecl:
		d.Specs = specListExports(d.Specs)
		return len(d.Specs) > 0
	case *FuncDecl:
		d.Body = nil // strip body
		return d.Name.IsExported()
	}
	return false
}


// FileExports trims the AST for a Go source file in place such that only
// exported nodes remain: all top-level identifiers which are not exported
// and their associated information (such as type, initial value, or function
// body) are removed. Non-exported fields and methods of exported types are
// stripped, and the function bodies of exported functions are set to nil.
// The File.comments list is not changed.
//
// FileExports returns true if there is an exported declaration; it returns
// false otherwise.
//
func FileExports(src *File) bool {
	j := 0
	for _, d := range src.Decls {
		if declExports(d) {
			src.Decls[j] = d
			j++
		}
	}
	src.Decls = src.Decls[0:j]
	return j > 0
}


// PackageExports trims the AST for a Go package in place such that only
// exported nodes remain. The pkg.Files list is not changed, so that file
// names and top-level package comments don't get lost.
//
// PackageExports returns true if there is an exported declaration; it
// returns false otherwise.
//
func PackageExports(pkg *Package) bool {
	hasExports := false
	for _, f := range pkg.Files {
		if FileExports(f) {
			hasExports = true
		}
	}
	return hasExports
}


// ----------------------------------------------------------------------------
// General filtering

type Filter func(string) bool

func filterIdentList(list []*Ident, f Filter) []*Ident {
	j := 0
	for _, x := range list {
		if f(x.Name) {
			list[j] = x
			j++
		}
	}
	return list[0:j]
}


func filterSpec(spec Spec, f Filter) bool {
	switch s := spec.(type) {
	case *ValueSpec:
		s.Names = filterIdentList(s.Names, f)
		return len(s.Names) > 0
	case *TypeSpec:
		return f(s.Name.Name)
	}
	return false
}


func filterSpecList(list []Spec, f Filter) []Spec {
	j := 0
	for _, s := range list {
		if filterSpec(s, f) {
			list[j] = s
			j++
		}
	}
	return list[0:j]
}


func filterDecl(decl Decl, f Filter) bool {
	switch d := decl.(type) {
	case *GenDecl:
		d.Specs = filterSpecList(d.Specs, f)
		return len(d.Specs) > 0
	case *FuncDecl:
		return f(d.Name.Name)
	}
	return false
}


// FilterFile trims the AST for a Go file in place by removing all
// names from top-level declarations (but not from parameter lists
// or inside types) that don't pass through the filter f. If the
// declaration is empty afterwards, the declaration is removed from
// the AST.
// The File.comments list is not changed.
//
// FilterFile returns true if there are any top-level declarations
// left after filtering; it returns false otherwise.
//
func FilterFile(src *File, f Filter) bool {
	j := 0
	for _, d := range src.Decls {
		if filterDecl(d, f) {
			src.Decls[j] = d
			j++
		}
	}
	src.Decls = src.Decls[0:j]
	return j > 0
}


// FilterPackage trims the AST for a Go package in place by removing all
// names from top-level declarations (but not from parameter lists
// or inside types) that don't pass through the filter f. If the
// declaration is empty afterwards, the declaration is removed from
// the AST.
// The pkg.Files list is not changed, so that file names and top-level
// package comments don't get lost.
//
// FilterPackage returns true if there are any top-level declarations
// left after filtering; it returns false otherwise.
//
func FilterPackage(pkg *Package, f Filter) bool {
	hasDecls := false
	for _, src := range pkg.Files {
		if FilterFile(src, f) {
			hasDecls = true
		}
	}
	return hasDecls
}


// ----------------------------------------------------------------------------
// Merging of package files

// The MergeMode flags control the behavior of MergePackageFiles.
type MergeMode uint

const (
	// If set, duplicate function declarations are excluded.
	FilterFuncDuplicates MergeMode = 1 << iota
	// If set, comments that are not associated with a specific
	// AST node (as Doc or Comment) are excluded.
	FilterUnassociatedComments
)

// separator is an empty //-style comment that is interspersed between
// different comment groups when they are concatenated into a single group
//
var separator = &Comment{noPos, []byte("//")}


// MergePackageFiles creates a file AST by merging the ASTs of the
// files belonging to a package. The mode flags control merging behavior.
//
func MergePackageFiles(pkg *Package, mode MergeMode) *File {
	// Count the number of package docs, comments and declarations across
	// all package files.
	ndocs := 0
	ncomments := 0
	ndecls := 0
	for _, f := range pkg.Files {
		if f.Doc != nil {
			ndocs += len(f.Doc.List) + 1 // +1 for separator
		}
		ncomments += len(f.Comments)
		ndecls += len(f.Decls)
	}

	// Collect package comments from all package files into a single
	// CommentGroup - the collected package documentation. The order
	// is unspecified. In general there should be only one file with
	// a package comment; but it's better to collect extra comments
	// than drop them on the floor.
	var doc *CommentGroup
	var pos token.Pos
	if ndocs > 0 {
		list := make([]*Comment, ndocs-1) // -1: no separator before first group
		i := 0
		for _, f := range pkg.Files {
			if f.Doc != nil {
				if i > 0 {
					// not the first group - add separator
					list[i] = separator
					i++
				}
				for _, c := range f.Doc.List {
					list[i] = c
					i++
				}
				if f.Package > pos {
					// Keep the maximum package clause position as
					// position for the package clause of the merged
					// files.
					pos = f.Package
				}
			}
		}
		doc = &CommentGroup{list}
	}

	// Collect declarations from all package files.
	var decls []Decl
	if ndecls > 0 {
		decls = make([]Decl, ndecls)
		funcs := make(map[string]int) // map of global function name -> decls index
		i := 0                        // current index
		n := 0                        // number of filtered entries
		for _, f := range pkg.Files {
			for _, d := range f.Decls {
				if mode&FilterFuncDuplicates != 0 {
					// A language entity may be declared multiple
					// times in different package files; only at
					// build time declarations must be unique.
					// For now, exclude multiple declarations of
					// functions - keep the one with documentation.
					//
					// TODO(gri): Expand this filtering to other
					//            entities (const, type, vars) if
					//            multiple declarations are common.
					if f, isFun := d.(*FuncDecl); isFun {
						name := f.Name.Name
						if j, exists := funcs[name]; exists {
							// function declared already
							if decls[j] != nil && decls[j].(*FuncDecl).Doc == nil {
								// existing declaration has no documentation;
								// ignore the existing declaration
								decls[j] = nil
							} else {
								// ignore the new declaration
								d = nil
							}
							n++ // filtered an entry
						} else {
							funcs[name] = i
						}
					}
				}
				decls[i] = d
				i++
			}
		}

		// Eliminate nil entries from the decls list if entries were
		// filtered. We do this using a 2nd pass in order to not disturb
		// the original declaration order in the source (otherwise, this
		// would also invalidate the monotonically increasing position
		// info within a single file).
		if n > 0 {
			i = 0
			for _, d := range decls {
				if d != nil {
					decls[i] = d
					i++
				}
			}
			decls = decls[0:i]
		}
	}

	// Collect comments from all package files.
	var comments []*CommentGroup
	if mode&FilterUnassociatedComments == 0 {
		comments = make([]*CommentGroup, ncomments)
		i := 0
		for _, f := range pkg.Files {
			i += copy(comments[i:], f.Comments)
		}
	}

	return &File{doc, pos, NewIdent(pkg.Name), decls, comments}
}
