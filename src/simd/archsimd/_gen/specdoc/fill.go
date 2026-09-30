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
// When rewriting a declaration's doc comment, [Fill] replaces existing
// spec-owned paragraphs with the corresponding spec documentation, while
// preserving existing implementation notes and directives unchanged. Because
// spec documentation is guaranteed never to begin with an implementation note
// prefix, the rewrite operation is idempotent: running [Fill] repeatedly on
// rewritten source produces byte-identical output.
package specdoc

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"simd/archsimd/_gen/specgen"
	"strings"
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

	// Filename optionally provides the name of the file being processed,
	// used for positions in report diagnostics.
	Filename string
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

	var report Report

	for _, d := range file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || !fd.Name.IsExported() {
			continue
		}

		decl := Decl{
			Pos:  fset.Position(fd.Pos()),
			Recv: recvTypeName(fd.Recv),
			Name: fd.Name.Name,
		}

		// Find the spec function
		specFn := spec.Lookup(decl.Recv, decl.Name)
		if specFn == nil {
			if opts.RejectUnknown {
				report.UnknownDecls = append(report.UnknownDecls, UnknownDecl{decl})
			}
			continue
		}

		// Now we know this is a spec API function and not some other one-off
		// that might not even be representable in spec.
		//
		// Parse the API declaration into a specgen.Func.

		doc := parseCommentGroup(fd.Doc)
		if !opts.AllowDocRewrite {
			for _, p := range doc {
				if p.Kind == specgen.SpecOwned {
					report.UnexpectedDocs = append(report.UnexpectedDocs, UnexpectedDoc{decl})
					break
				}
			}
		}

		var recv specgen.Arg
		if fd.Recv != nil && len(fd.Recv.List) == 1 {
			recvs, err := fieldListArgs(fd.Recv)
			if err != nil {
				return src, fmt.Errorf("%s: %s receiver: %w", decl.Pos, decl.Name, err)
			}
			recv = recvs[0]
		}

		in, err := fieldListArgs(fd.Type.Params)
		if err != nil {
			return src, fmt.Errorf("%s: %s params: %w", decl.Pos, decl.Name, err)
		}
		out, err := fieldListArgs(fd.Type.Results)
		if err != nil {
			return src, fmt.Errorf("%s: %s results: %w", decl.Pos, decl.Name, err)
		}

		// Create a skeleton Func for the actual declaration
		declFn := &specgen.Func{Doc: doc, Recv: recv, Name: decl.Name, In: in, Out: out}

		// Check comment section ordering
		if len(declFn.Doc) > 0 {
			if _, orderErr := specgen.FormatComment(doc); orderErr != nil {
				report.DocOrderViolations = append(report.DocOrderViolations, DocOrderViolation{
					D:   decl,
					Err: orderErr,
				})
			}
		}

		// Compare function signatures
		nameDetails, typeDetails := compareFuncs(declFn, specFn)
		if len(nameDetails) > 0 && !opts.AllowNameMismatches {
			report.NameMismatches = append(report.NameMismatches, NameMismatch{
				D:       decl,
				DeclSig: declFn.Signature(),
				SpecSig: specFn.Signature(),
				Details: nameDetails,
			})
		}
		if len(typeDetails) > 0 {
			report.TypeMismatches = append(report.TypeMismatches, TypeMismatch{
				D:       decl,
				DeclSig: declFn.Signature(),
				SpecSig: specFn.Signature(),
				Details: typeDetails,
			})
		}
	}

	if report.Empty() {
		return src, nil
	}
	return src, &report
}

func namedArg(arg specgen.Arg) bool {
	return !(arg.Name == "" || arg.Name == "_")
}

// compareFuncs compares the receiver, parameters, and results of declFn against
// specFn and returns details of any mismatches found.
func compareFuncs(declFn, specFn *specgen.Func) (nameDetails, typeDetails []string) {
	if (declFn.Recv.Type != nil) != (specFn.Recv.Type != nil) {
		// This should never happen because we look up by receiver.
		panic("cannot compare function and method")
	}

	compareArg := func(decl, spec specgen.Arg, label string, args ...any) {
		if namedArg(spec) && decl.Name != spec.Name {
			var buf strings.Builder
			fmt.Fprintf(&buf, label, args...)
			fmt.Fprintf(&buf, ": API name %q != spec name %q", decl.Name, spec.Name)
			nameDetails = append(nameDetails, buf.String())
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
