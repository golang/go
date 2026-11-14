// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package specdoc validates exported SIMD declarations against package
// simd/internal/spec and manages the composition and injection of spec-derived
// documentation into Go source files.
//
// Generators and tools pass Go source buffers through [Fill] before writing
// them to disk. Comparing generated or hand-written declarations against spec
// ensures that the API implementation and the specification agree on types and
// parameter names, and allows shared documentation to be maintained in a single
// place.
//
// # Doc-Ownership Convention
//
// Doc comments follow a three-part structural convention:
//
// 1. Spec-owned paragraphs: Describe the behavioral semantics of the operation.
// These are managed by specdoc and replaced with documentation from spec when a
// matching spec entry exists.
//
// 2. Implementation notes: Describe target-specific implementation details,
// such as assembly instructions, required CPU features, emulation notes,
// performance caveats, or deprecations. An implementation note is any paragraph
// whose first line begins with one of a fixed set of prefixes: "Asm:", "CPU
// Feature:", "Emulated", "Deprecated:", "Performance:", or "Note:".
//
// 3. Directive comments: Tool and compiler directives, such as //go:noescape.
//
// These sections must appear strictly in this order: spec-owned paragraphs
// strictly before implementation notes, which must appear strictly before
// directive comments.
//
// # Rewriting and Idempotence
//
// When rewriting a declaration, [Fill] replaces existing spec-owned paragraphs
// with the corresponding spec documentation (unless [Options.NoFillDoc] is set)
// and fills unnamed parameters and results with canonical names from spec
// (unless [Options.NoFillNames] is set), while preserving existing
// implementation notes and directives unchanged. Because spec documentation is
// guaranteed never to begin with an implementation note prefix, the rewrite
// operation is idempotent: running [Fill] repeatedly on rewritten source
// produces byte-identical output.
package specdoc

import (
	"bytes"
	"cmp"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"

	"simd/archsimd/_gen/specgen"
)

// Options configures the behavior of [Fill].
type Options struct {
	// RejectUnknown controls whether an exported declaration with no matching
	// spec entry is reported as an error.
	//
	// When false, un-specced declarations are tolerated and omitted from
	// signature and doc checks, and passed through the rewrite verbatim. When
	// true, any exported declaration lacking a spec definition causes
	// generation to fail, preventing new API operations from escaping spec
	// coverage.
	RejectUnknown bool

	// AllowDocRewrite controls whether an exported declaration containing
	// existing spec-owned doc comments is tolerated.
	//
	// When false (default), any spec-owned doc comment found in the input is
	// reported as an error (recorded in [Report.UnexpectedDocs]). This is used
	// by generators to ensure they do not emit doc comments that would be
	// overwritten by spec. Implementation notes and directives are still
	// permitted.
	//
	// When true, existing spec-owned doc comments are permitted and rewritten.
	// This is used when processing hand-written source files.
	AllowDocRewrite bool

	// AllowNameMismatches controls whether parameter and result name mismatches
	// between an AST declaration and spec are tolerated.
	//
	// When false (default), name mismatches are reported as errors.
	// When true, name mismatches are omitted from the report.
	AllowNameMismatches bool

	// NoFillDoc disables replacing doc comments with spec-derived comments.
	//
	// When false, Fill updates spec-owned doc paragraphs.
	NoFillDoc bool

	// NoFillNames disables filling unnamed parameters and results with names
	// from spec.
	//
	// When false, Fill fills unnamed parameters and results in exported
	// declarations from the matching spec function.
	NoFillNames bool

	// Filename optionally provides the name of the file being processed,
	// used for positions in report diagnostics.
	Filename string
}

type textEdit struct {
	start int
	end   int
	text  string
}

// Filler returns a post-processing hook that runs [Fill] on generated Go source files.
// The returned function satisfies [gentools.PostProcessor].
func Filler(idx *specgen.Index, opts Options) func(relPath string, isGo bool, content []byte) ([]byte, error) {
	return func(relPath string, isGo bool, src []byte) ([]byte, error) {
		if !isGo {
			return src, nil
		}
		dir := filepath.ToSlash(filepath.Dir(relPath))
		if dir != "simd" && dir != "simd/archsimd" {
			return src, nil
		}
		fileOpts := opts
		fileOpts.Filename = filepath.ToSlash(relPath)
		return Fill(src, idx, fileOpts)
	}
}

// Fill parses Go source code, extracts exported declarations, verifies their
// signatures and doc comments against spec, and returns the modified source.
// If any verification checks fail, Fill returns a [*Report] error containing
// all findings.
func Fill(src []byte, spec *specgen.Index, opts Options) ([]byte, error) {
	fset := token.NewFileSet()
	filename := opts.Filename
	if filename == "" {
		filename = "<unknown>"
	}
	file, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
	if err != nil {
		return src, fmt.Errorf("parsing %s: %w", filename, err)
	}

	f := &filler{fset: fset, src: src, spec: spec, opts: opts}
	for _, d := range file.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok {
			if err := f.fillFunc(fd); err != nil {
				return src, err
			}
		}
	}
	report, edits := f.report, f.edits

	var repErr error
	if !report.Empty() {
		repErr = &report
	}

	if len(edits) == 0 {
		return src, repErr
	}

	rewritten := applyEdits(src, edits)
	formatted, err := format.Source(rewritten)
	if err != nil {
		if repErr != nil {
			return rewritten, repErr
		}
		return rewritten, fmt.Errorf("formatting source: %w", err)
	}
	return formatted, repErr
}

// filler holds the state of a single call to [Fill].
type filler struct {
	fset *token.FileSet
	src  []byte
	spec *specgen.Index
	opts Options

	report Report
	edits  []textEdit
}

// fillFunc checks the function or method declaration fd against spec and
// records the edits to fill it in.
func (f *filler) fillFunc(fd *ast.FuncDecl) error {
	if !fd.Name.IsExported() {
		return nil
	}

	decl := Decl{
		Pos:  f.fset.Position(fd.Pos()),
		Recv: recvTypeName(fd.Recv),
		Name: fd.Name.Name,
	}

	// Find the spec function
	specFn := f.spec.Lookup(decl.Recv, decl.Name)
	if specFn == nil {
		if f.opts.RejectUnknown {
			f.report.UnknownDecls = append(f.report.UnknownDecls, UnknownDecl{decl})
		}
		return nil
	}

	// Now we know this is a spec API function and not some other one-off
	// that might not even be representable in spec.

	// Check and fill the doc comment
	f.fillDoc(decl, fd.Doc, specFn.Doc)

	// Parse the API declaration into a specgen.Func.
	var recv specgen.Arg
	if fd.Recv != nil && len(fd.Recv.List) == 1 {
		recvs, err := fieldListArgs(fd.Recv)
		if err != nil {
			return fmt.Errorf("%s: %s receiver: %w", decl.Pos, decl.Name, err)
		}
		recv = recvs[0]
	}

	in, err := fieldListArgs(fd.Type.Params)
	if err != nil {
		return fmt.Errorf("%s: %s params: %w", decl.Pos, decl.Name, err)
	}
	out, err := fieldListArgs(fd.Type.Results)
	if err != nil {
		return fmt.Errorf("%s: %s results: %w", decl.Pos, decl.Name, err)
	}

	// Create a skeleton Func for the actual declaration
	declFn := &specgen.Func{Recv: recv, Name: decl.Name, In: in, Out: out}

	// Compare function signatures
	nameDetails, typeDetails := compareFuncs(declFn, specFn, f.opts.NoFillNames)
	if len(nameDetails) > 0 && !f.opts.AllowNameMismatches {
		f.report.NameMismatches = append(f.report.NameMismatches, NameMismatch{
			D:       decl,
			DeclSig: declFn.Signature(),
			SpecSig: specFn.Signature(),
			Details: nameDetails,
		})
	}
	if len(typeDetails) > 0 {
		f.report.TypeMismatches = append(f.report.TypeMismatches, TypeMismatch{
			D:       decl,
			DeclSig: declFn.Signature(),
			SpecSig: specFn.Signature(),
			Details: typeDetails,
		})
	}

	// Rewrite parameter and result names
	if !f.opts.NoFillNames && len(nameDetails) == 0 && len(typeDetails) == 0 {
		if edit, ok := rewriteSignature(f.fset, f.src, fd, declFn, specFn); ok {
			f.edits = append(f.edits, edit)
		}
	}
	return nil
}

func namedArg(arg specgen.Arg) bool {
	return !(arg.Name == "" || arg.Name == "_")
}

// compareFuncs compares the receiver, parameters, and results of declFn against
// specFn and returns details of any mismatches found.
func compareFuncs(declFn, specFn *specgen.Func, noFillNames bool) (nameDetails, typeDetails []string) {
	if (declFn.Recv.Type != nil) != (specFn.Recv.Type != nil) {
		// This should never happen because we look up by receiver.
		panic("cannot compare function and method")
	}

	compareArg := func(decl, spec specgen.Arg, label string, args ...any) {
		if namedArg(spec) && decl.Name != spec.Name {
			if !namedArg(decl) && !noFillNames {
				// An unnamed parameter or result is an opportunity to fill
				// from spec, not an error, when name filling is active.
			} else {
				var buf strings.Builder
				fmt.Fprintf(&buf, label, args...)
				fmt.Fprintf(&buf, ": API name %q != spec name %q", decl.Name, spec.Name)
				nameDetails = append(nameDetails, buf.String())
			}
		}
		if decl.Type != spec.Type {
			var buf strings.Builder
			fmt.Fprintf(&buf, label, args...)
			if spec.Name != "" {
				fmt.Fprintf(&buf, " (%s)", spec.Name)
			}
			fmt.Fprintf(&buf, ": API type %s != spec type %s", decl.Type, spec.Type)
			typeDetails = append(typeDetails, buf.String())
		}
	}
	compareArgs := func(decl, spec []specgen.Arg, label string) {
		if len(decl) != len(spec) {
			typeDetails = append(typeDetails, fmt.Sprintf("%s count mismatch: API has %d, spec has %d", label, len(decl), len(spec)))
			return
		}
		for i := range decl {
			compareArg(decl[i], spec[i], "%s %d", label, i)
		}
	}

	if specFn.Recv.Type != nil {
		compareArg(declFn.Recv, specFn.Recv, "receiver")
	}
	compareArgs(declFn.In, specFn.In, "parameter")
	compareArgs(declFn.Out, specFn.Out, "result")

	return nameDetails, typeDetails
}

// recvTypeName extracts the receiver type of a method. It returns "" for
// package-level functions (nil or empty FieldList).
func recvTypeName(fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	t := fl.List[0].Type
	for {
		switch tt := t.(type) {
		case *ast.Ident:
			return tt.Name

		case *ast.StarExpr:
			t = tt.X
		case *ast.ParenExpr:
			t = tt.X
		case *ast.IndexExpr:
			t = tt.X
		case *ast.IndexListExpr:
			t = tt.X

		default:
			return "<unknown>"
		}
	}
}

// fillDoc checks the doc comment docGroup of decl, recording any findings.
// Unless opts.NoFillDoc is set, it then records an edit that replaces
// docGroup's spec-owned paragraphs with specDoc, keeping existing
// implementation notes and directives. If docGroup is nil, the new comment is
// inserted at decl.Pos.
func (f *filler) fillDoc(decl Decl, docGroup *ast.CommentGroup, specDoc []specgen.Paragraph) {
	doc := parseCommentGroup(docGroup)
	if !f.opts.AllowDocRewrite {
		for _, p := range doc {
			if p.Kind == specgen.SpecOwned {
				f.report.UnexpectedDocs = append(f.report.UnexpectedDocs, UnexpectedDoc{decl})
				break
			}
		}
	}

	// Check comment section ordering
	if len(doc) > 0 {
		if _, err := specgen.FormatComment(doc); err != nil {
			f.report.DocOrderViolations = append(f.report.DocOrderViolations, DocOrderViolation{
				D:   decl,
				Err: err,
			})
			return
		}
	}

	if f.opts.NoFillDoc {
		return
	}

	// Compose replacement comment
	newParas := append([]specgen.Paragraph(nil), specDoc...)
	for _, p := range doc {
		if p.Kind == specgen.ImplementationNote || p.Kind == specgen.DirectiveComment {
			newParas = append(newParas, p)
		}
	}

	replacementComment, err := specgen.FormatComment(newParas)
	if err != nil {
		return
	}

	if docGroup != nil {
		// Replace existing doc.
		start := f.fset.Position(docGroup.Pos()).Offset
		end := f.fset.Position(docGroup.End()).Offset + 1 // Include final \n
		if string(f.src[start:end]) != replacementComment {
			f.edits = append(f.edits, textEdit{start: start, end: end, text: replacementComment})
		}
	} else if replacementComment != "" {
		// Pure insertion.
		start := decl.Pos.Offset
		f.edits = append(f.edits, textEdit{start: start, end: start, text: replacementComment})
	}
}

// rewriteSignature computes the textEdit to fill unnamed parameter and result
// names in fd.
func rewriteSignature(fset *token.FileSet, src []byte, fd *ast.FuncDecl, declFn, specFn *specgen.Func) (textEdit, bool) {
	// Create a synthetic specgen.Func with merged arguments.
	mergedFn := &specgen.Func{
		Recv: declFn.Recv,
		Name: declFn.Name,
		In:   mergeArgs(declFn.In, specFn.In),
		Out:  mergeArgs(declFn.Out, specFn.Out),
	}
	newSig := mergedFn.Signature()
	start := fset.Position(fd.Pos()).Offset
	end := fset.Position(fd.Type.End()).Offset
	if string(src[start:end]) != newSig {
		return textEdit{start: start, end: end, text: newSig}, true
	}
	return textEdit{}, false
}

func mergeArgs(declArgs, specArgs []specgen.Arg) []specgen.Arg {
	merged := slices.Clone(declArgs)
	// Fill any unnamed arguments from spec
	for i := range merged {
		if namedArg(specArgs[i]) && !namedArg(merged[i]) {
			merged[i].Name = specArgs[i].Name
		}
	}
	return merged
}

// applyEdits applies text edits to src in ascending offset order and returns
// the modified byte slice.
func applyEdits(src []byte, edits []textEdit) []byte {
	if len(edits) == 0 {
		return src
	}

	// Apply edits in ascending offset order
	slices.SortFunc(edits, func(a, b textEdit) int {
		return cmp.Compare(a.start, b.start)
	})

	var out bytes.Buffer
	last := 0
	for _, e := range edits {
		out.Write(src[last:e.start])
		out.WriteString(e.text)
		last = e.end
	}
	out.Write(src[last:])
	return out.Bytes()
}
