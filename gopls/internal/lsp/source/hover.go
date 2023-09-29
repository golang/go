// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/constant"
	"go/doc"
	"go/format"
	"go/token"
	"go/types"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"golang.org/x/text/unicode/runenames"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/tokeninternal"
	"golang.org/x/tools/internal/typeparams"
)

// HoverJSON contains information used by hover. It is also the JSON returned
// for the "structured" hover format
type HoverJSON struct {
	// Synopsis is a single sentence synopsis of the symbol's documentation.
	Synopsis string `json:"synopsis"`

	// FullDocumentation is the symbol's full documentation.
	FullDocumentation string `json:"fullDocumentation"`

	// Signature is the symbol's signature.
	Signature string `json:"signature"`

	// SingleLine is a single line describing the symbol.
	// This is recommended only for use in clients that show a single line for hover.
	SingleLine string `json:"singleLine"`

	// SymbolName is the human-readable name to use for the symbol in links.
	SymbolName string `json:"symbolName"`

	// LinkPath is the pkg.go.dev link for the given symbol.
	// For example, the "go/ast" part of "pkg.go.dev/go/ast#Node".
	LinkPath string `json:"linkPath"`

	// LinkAnchor is the pkg.go.dev link anchor for the given symbol.
	// For example, the "Node" part of "pkg.go.dev/go/ast#Node".
	LinkAnchor string `json:"linkAnchor"`
}

// Hover implements the "textDocument/hover" RPC for Go files.
func Hover(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) (*protocol.Hover, error) {
	ctx, done := event.Start(ctx, "source.Hover")
	defer done()

	rng, h, err := hover(ctx, snapshot, fh, position)
	if err != nil {
		return nil, err
	}
	if h == nil {
		return nil, nil
	}
	hover, err := formatHover(h, snapshot.Options())
	if err != nil {
		return nil, err
	}
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  snapshot.Options().PreferredContentFormat,
			Value: hover,
		},
		Range: rng,
	}, nil
}

// hover computes hover information at the given position. If we do not support
// hovering at the position, it returns _, nil, nil: an error is only returned
// if the position is valid but we fail to compute hover information.
func hover(ctx context.Context, snapshot Snapshot, fh FileHandle, pp protocol.Position) (protocol.Range, *HoverJSON, error) {
	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return protocol.Range{}, nil, err
	}
	pos, err := pgf.PositionPos(pp)
	if err != nil {
		return protocol.Range{}, nil, err
	}

	// Handle hovering over import paths, which do not have an associated
	// identifier.
	for _, spec := range pgf.File.Imports {
		// We are inclusive of the end point here to allow hovering when the cursor
		// is just after the import path.
		if spec.Path.Pos() <= pos && pos <= spec.Path.End() {
			return hoverImport(ctx, snapshot, pkg, pgf, spec)
		}
	}

	// Handle hovering over the package name, which does not have an associated
	// object.
	// As with import paths, we allow hovering just after the package name.
	if pgf.File.Name != nil && pgf.File.Name.Pos() <= pos && pos <= pgf.File.Name.Pos() {
		return hoverPackageName(pkg, pgf)
	}

	// Handle hovering over (non-import-path) literals.
	if path, _ := astutil.PathEnclosingInterval(pgf.File, pos, pos); len(path) > 0 {
		if lit, _ := path[0].(*ast.BasicLit); lit != nil {
			return hoverLit(pgf, lit, pos)
		}
	}

	// Handle hovering over embed directive argument.
	pattern, embedRng := parseEmbedDirective(pgf.Mapper, pp)
	if pattern != "" {
		return hoverEmbed(fh, embedRng, pattern)
	}

	// Handle linkname directive by overriding what to look for.
	var linkedRange *protocol.Range // range referenced by linkname directive, or nil
	if pkgPath, name, offset := parseLinkname(ctx, snapshot, fh, pp); pkgPath != "" && name != "" {
		// rng covering 2nd linkname argument: pkgPath.name.
		rng, err := pgf.PosRange(pgf.Tok.Pos(offset), pgf.Tok.Pos(offset+len(pkgPath)+len(".")+len(name)))
		if err != nil {
			return protocol.Range{}, nil, fmt.Errorf("range over linkname arg: %w", err)
		}
		linkedRange = &rng

		pkg, pgf, pos, err = findLinkname(ctx, snapshot, PackagePath(pkgPath), name)
		if err != nil {
			return protocol.Range{}, nil, fmt.Errorf("find linkname: %w", err)
		}
	}

	// The general case: compute hover information for the object referenced by
	// the identifier at pos.
	ident, obj, selectedType := referencedObject(pkg, pgf, pos)
	if obj == nil || ident == nil {
		return protocol.Range{}, nil, nil // no object to hover
	}

	// Unless otherwise specified, rng covers the ident being hovered.
	var rng protocol.Range
	if linkedRange != nil {
		rng = *linkedRange
	} else {
		rng, err = pgf.NodeRange(ident)
		if err != nil {
			return protocol.Range{}, nil, err
		}
	}

	// By convention, we qualify hover information relative to the package
	// from which the request originated.
	qf := Qualifier(pgf.File, pkg.GetTypes(), pkg.GetTypesInfo())

	// Handle type switch identifiers as a special case, since they don't have an
	// object.
	//
	// There's not much useful information to provide.
	if selectedType != nil {
		fakeObj := types.NewVar(obj.Pos(), obj.Pkg(), obj.Name(), selectedType)
		signature := types.ObjectString(fakeObj, qf)
		return rng, &HoverJSON{
			Signature:  signature,
			SingleLine: signature,
			SymbolName: fakeObj.Name(),
		}, nil
	}

	// Handle builtins, which don't have a package or position.
	if obj.Pkg() == nil {
		h, err := hoverBuiltin(ctx, snapshot, obj)
		return rng, h, err
	}

	// For all other objects, consider the full syntax of their declaration in
	// order to correctly compute their documentation, signature, and link.
	declPGF, declPos, err := parseFull(ctx, snapshot, pkg.FileSet(), obj.Pos())
	if err != nil {
		return protocol.Range{}, nil, fmt.Errorf("re-parsing declaration of %s: %v", obj.Name(), err)
	}
	decl, spec, field := findDeclInfo([]*ast.File{declPGF.File}, declPos)
	comment := chooseDocComment(decl, spec, field)
	docText := comment.Text()

	// By default, types.ObjectString provides a reasonable signature.
	signature := objectString(obj, qf, declPos, declPGF.Tok, spec)
	singleLineSignature := signature

	// TODO(rfindley): we could do much better for inferred signatures.
	if inferred := inferredSignature(pkg.GetTypesInfo(), ident); inferred != nil {
		if s := inferredSignatureString(obj, qf, inferred); s != "" {
			signature = s
		}
	}

	// For "objects defined by a type spec", the signature produced by
	// objectString is insufficient:
	//  (1) large structs are formatted poorly, with no newlines
	//  (2) we lose inline comments
	//
	// Furthermore, we include a summary of their method set.
	//
	// TODO(rfindley): this should use FormatVarType to get proper qualification
	// of identifiers, and we should revisit the formatting of method set.
	_, isTypeName := obj.(*types.TypeName)
	_, isTypeParam := obj.Type().(*typeparams.TypeParam)
	if isTypeName && !isTypeParam {
		spec, ok := spec.(*ast.TypeSpec)
		if !ok {
			return protocol.Range{}, nil, bug.Errorf("type name %q without type spec", obj.Name())
		}
		spec2 := *spec
		// Don't duplicate comments when formatting type specs.
		spec2.Doc = nil
		spec2.Comment = nil
		var b strings.Builder
		b.WriteString("type ")
		fset := tokeninternal.FileSetFor(declPGF.Tok)
		if err := format.Node(&b, fset, &spec2); err != nil {
			return protocol.Range{}, nil, err
		}

		// Display the declared methods accessible from the identifier.
		//
		// (The format.Node call above displays any struct fields, public
		// or private, in syntactic form. We choose not to recursively
		// enumerate any fields and methods promoted from them.)
		if !types.IsInterface(obj.Type()) {
			sep := "\n\n"
			for _, m := range typeutil.IntuitiveMethodSet(obj.Type(), nil) {
				// Show direct methods that are either exported, or defined in the
				// current package.
				if (m.Obj().Exported() || m.Obj().Pkg() == pkg.GetTypes()) && len(m.Index()) == 1 {
					b.WriteString(sep)
					sep = "\n"
					b.WriteString(types.ObjectString(m.Obj(), qf))
				}
			}
		}
		signature = b.String()
	}

	// Compute link data (on pkg.go.dev or other documentation host).
	//
	// If linkPath is empty, the symbol is not linkable.
	var (
		linkName string    // => link title, always non-empty
		linkPath string    // => link path
		anchor   string    // link anchor
		linkMeta *Metadata // metadata for the linked package
	)
	{
		linkMeta = findFileInDeps(snapshot, pkg.Metadata(), declPGF.URI)
		if linkMeta == nil {
			return protocol.Range{}, nil, bug.Errorf("no metadata for %s", declPGF.URI)
		}

		// For package names, we simply link to their imported package.
		if pkgName, ok := obj.(*types.PkgName); ok {
			linkName = pkgName.Name()
			linkPath = pkgName.Imported().Path()
			impID := linkMeta.DepsByPkgPath[PackagePath(pkgName.Imported().Path())]
			linkMeta = snapshot.Metadata(impID)
			if linkMeta == nil {
				return protocol.Range{}, nil, bug.Errorf("no metadata for %s", declPGF.URI)
			}
		} else {
			// For all others, check whether the object is in the package scope, or
			// an exported field or method of an object in the package scope.
			//
			// We try to match pkgsite's heuristics for what is linkable, and what is
			// not.
			var recv types.Object
			switch obj := obj.(type) {
			case *types.Func:
				sig := obj.Type().(*types.Signature)
				if sig.Recv() != nil {
					tname := typeToObject(sig.Recv().Type())
					if tname != nil { // beware typed nil
						recv = tname
					}
				}
			case *types.Var:
				if obj.IsField() {
					if spec, ok := spec.(*ast.TypeSpec); ok {
						typeName := spec.Name
						scopeObj, _ := obj.Pkg().Scope().Lookup(typeName.Name).(*types.TypeName)
						if scopeObj != nil {
							if st, _ := scopeObj.Type().Underlying().(*types.Struct); st != nil {
								for i := 0; i < st.NumFields(); i++ {
									if obj == st.Field(i) {
										recv = scopeObj
									}
								}
							}
						}
					}
				}
			}

			// Even if the object is not available in package documentation, it may
			// be embedded in a documented receiver. Detect this by searching
			// enclosing selector expressions.
			//
			// TODO(rfindley): pkgsite doesn't document fields from embedding, just
			// methods.
			if recv == nil || !recv.Exported() {
				path := pathEnclosingObjNode(pgf.File, pos)
				if enclosing := searchForEnclosing(pkg.GetTypesInfo(), path); enclosing != nil {
					recv = enclosing
				} else {
					recv = nil // note: just recv = ... could result in a typed nil.
				}
			}

			pkg := obj.Pkg()
			if recv != nil {
				linkName = fmt.Sprintf("(%s.%s).%s", pkg.Name(), recv.Name(), obj.Name())
				if obj.Exported() && recv.Exported() && pkg.Scope().Lookup(recv.Name()) == recv {
					linkPath = pkg.Path()
					anchor = fmt.Sprintf("%s.%s", recv.Name(), obj.Name())
				}
			} else {
				linkName = fmt.Sprintf("%s.%s", pkg.Name(), obj.Name())
				if obj.Exported() && pkg.Scope().Lookup(obj.Name()) == obj {
					linkPath = pkg.Path()
					anchor = obj.Name()
				}
			}
		}
	}

	if snapshot.View().IsGoPrivatePath(linkPath) || linkMeta.ForTest != "" {
		linkPath = ""
	} else if linkMeta.Module != nil && linkMeta.Module.Version != "" {
		mod := linkMeta.Module
		linkPath = strings.Replace(linkPath, mod.Path, mod.Path+"@"+mod.Version, 1)
	}

	return rng, &HoverJSON{
		Synopsis:          doc.Synopsis(docText),
		FullDocumentation: docText,
		SingleLine:        singleLineSignature,
		SymbolName:        linkName,
		Signature:         signature,
		LinkPath:          linkPath,
		LinkAnchor:        anchor,
	}, nil
}

// hoverBuiltin computes hover information when hovering over a builtin
// identifier.
func hoverBuiltin(ctx context.Context, snapshot Snapshot, obj types.Object) (*HoverJSON, error) {
	// TODO(rfindley): link to the correct version of Go documentation.
	builtin, err := snapshot.BuiltinFile(ctx)
	if err != nil {
		return nil, err
	}

	if obj.Name() == "Error" {
		signature := obj.String()
		return &HoverJSON{
			Signature:  signature,
			SingleLine: signature,
			// TODO(rfindley): these are better than the current behavior.
			// SymbolName: "(error).Error",
			// LinkPath:   "builtin",
			// LinkAnchor: "error.Error",
		}, nil
	}

	builtinObj := builtin.File.Scope.Lookup(obj.Name())
	if builtinObj == nil {
		// All builtins should have a declaration in the builtin file.
		return nil, bug.Errorf("no builtin object for %s", obj.Name())
	}
	node, _ := builtinObj.Decl.(ast.Node)
	if node == nil {
		return nil, bug.Errorf("no declaration for %s", obj.Name())
	}

	var comment *ast.CommentGroup
	path, _ := astutil.PathEnclosingInterval(builtin.File, node.Pos(), node.End())
	for _, n := range path {
		switch n := n.(type) {
		case *ast.GenDecl:
			// Separate documentation and signature.
			comment = n.Doc
			node2 := *n
			node2.Doc = nil
			node = &node2
		case *ast.FuncDecl:
			// Ditto.
			comment = n.Doc
			node2 := *n
			node2.Doc = nil
			node = &node2
		}
	}

	signature := FormatNodeFile(builtin.Tok, node)
	// Replace fake types with their common equivalent.
	// TODO(rfindley): we should instead use obj.Type(), which would have the
	// *actual* types of the builtin call.
	signature = replacer.Replace(signature)

	docText := comment.Text()
	return &HoverJSON{
		Synopsis:          doc.Synopsis(docText),
		FullDocumentation: docText,
		Signature:         signature,
		SingleLine:        obj.String(),
		SymbolName:        obj.Name(),
		LinkPath:          "builtin",
		LinkAnchor:        obj.Name(),
	}, nil
}

// hoverImport computes hover information when hovering over the import path of
// imp in the file pgf of pkg.
//
// If we do not have metadata for the hovered import, it returns _
func hoverImport(ctx context.Context, snapshot Snapshot, pkg Package, pgf *ParsedGoFile, imp *ast.ImportSpec) (protocol.Range, *HoverJSON, error) {
	rng, err := pgf.NodeRange(imp.Path)
	if err != nil {
		return protocol.Range{}, nil, err
	}

	importPath := UnquoteImportPath(imp)
	if importPath == "" {
		return protocol.Range{}, nil, fmt.Errorf("invalid import path")
	}
	impID := pkg.Metadata().DepsByImpPath[importPath]
	if impID == "" {
		return protocol.Range{}, nil, fmt.Errorf("no package data for import %q", importPath)
	}
	impMetadata := snapshot.Metadata(impID)
	if impMetadata == nil {
		return protocol.Range{}, nil, bug.Errorf("failed to resolve import ID %q", impID)
	}

	// Find the first file with a package doc comment.
	var comment *ast.CommentGroup
	for _, f := range impMetadata.CompiledGoFiles {
		fh, err := snapshot.ReadFile(ctx, f)
		if err != nil {
			if ctx.Err() != nil {
				return protocol.Range{}, nil, ctx.Err()
			}
			continue
		}
		pgf, err := snapshot.ParseGo(ctx, fh, ParseHeader)
		if err != nil {
			if ctx.Err() != nil {
				return protocol.Range{}, nil, ctx.Err()
			}
			continue
		}
		if pgf.File.Doc != nil {
			comment = pgf.File.Doc
			break
		}
	}

	docText := comment.Text()
	return rng, &HoverJSON{
		Synopsis:          doc.Synopsis(docText),
		FullDocumentation: docText,
	}, nil
}

// hoverPackageName computes hover information for the package name of the file
// pgf in pkg.
func hoverPackageName(pkg Package, pgf *ParsedGoFile) (protocol.Range, *HoverJSON, error) {
	var comment *ast.CommentGroup
	for _, pgf := range pkg.CompiledGoFiles() {
		if pgf.File.Doc != nil {
			comment = pgf.File.Doc
			break
		}
	}
	rng, err := pgf.NodeRange(pgf.File.Name)
	if err != nil {
		return protocol.Range{}, nil, err
	}
	docText := comment.Text()
	return rng, &HoverJSON{
		Synopsis:          doc.Synopsis(docText),
		FullDocumentation: docText,
		// Note: including a signature is redundant, since the cursor is already on the
		// package name.
	}, nil
}

// hoverLit computes hover information when hovering over the basic literal lit
// in the file pgf. The provided pos must be the exact position of the cursor,
// as it is used to extract the hovered rune in strings.
//
// For example, hovering over "\u2211" in "foo \u2211 bar" yields:
//
//	'∑', U+2211, N-ARY SUMMATION
func hoverLit(pgf *ParsedGoFile, lit *ast.BasicLit, pos token.Pos) (protocol.Range, *HoverJSON, error) {
	var (
		value      string    // if non-empty, a constant value to format in hover
		r          rune      // if non-zero, format a description of this rune in hover
		start, end token.Pos // hover span
	)
	// Extract a rune from the current position.
	// 'Ω', "...Ω...", or 0x03A9 => 'Ω', U+03A9, GREEK CAPITAL LETTER OMEGA
	switch lit.Kind {
	case token.CHAR:
		s, err := strconv.Unquote(lit.Value)
		if err != nil {
			// If the conversion fails, it's because of an invalid syntax, therefore
			// there is no rune to be found.
			return protocol.Range{}, nil, nil
		}
		r, _ = utf8.DecodeRuneInString(s)
		if r == utf8.RuneError {
			return protocol.Range{}, nil, fmt.Errorf("rune error")
		}
		start, end = lit.Pos(), lit.End()

	case token.INT:
		// Short literals (e.g. 99 decimal, 07 octal) are uninteresting.
		if len(lit.Value) < 3 {
			return protocol.Range{}, nil, nil
		}

		v := constant.MakeFromLiteral(lit.Value, lit.Kind, 0)
		if v.Kind() != constant.Int {
			return protocol.Range{}, nil, nil
		}

		switch lit.Value[:2] {
		case "0x", "0X":
			// As a special case, try to recognize hexadecimal literals as runes if
			// they are within the range of valid unicode values.
			if v, ok := constant.Int64Val(v); ok && v > 0 && v <= utf8.MaxRune && utf8.ValidRune(rune(v)) {
				r = rune(v)
			}
			fallthrough
		case "0o", "0O", "0b", "0B":
			// Format the decimal value of non-decimal literals.
			value = v.ExactString()
			start, end = lit.Pos(), lit.End()
		default:
			return protocol.Range{}, nil, nil
		}

	case token.STRING:
		// It's a string, scan only if it contains a unicode escape sequence under or before the
		// current cursor position.
		litOffset, err := safetoken.Offset(pgf.Tok, lit.Pos())
		if err != nil {
			return protocol.Range{}, nil, err
		}
		offset, err := safetoken.Offset(pgf.Tok, pos)
		if err != nil {
			return protocol.Range{}, nil, err
		}
		for i := offset - litOffset; i > 0; i-- {
			// Start at the cursor position and search backward for the beginning of a rune escape sequence.
			rr, _ := utf8.DecodeRuneInString(lit.Value[i:])
			if rr == utf8.RuneError {
				return protocol.Range{}, nil, fmt.Errorf("rune error")
			}
			if rr == '\\' {
				// Got the beginning, decode it.
				var tail string
				r, _, tail, err = strconv.UnquoteChar(lit.Value[i:], '"')
				if err != nil {
					// If the conversion fails, it's because of an invalid syntax,
					// therefore is no rune to be found.
					return protocol.Range{}, nil, nil
				}
				// Only the rune escape sequence part of the string has to be highlighted, recompute the range.
				runeLen := len(lit.Value) - (int(i) + len(tail))
				start = token.Pos(int(lit.Pos()) + int(i))
				end = token.Pos(int(start) + runeLen)
				break
			}
		}
	}

	if value == "" && r == 0 { // nothing to format
		return protocol.Range{}, nil, nil
	}

	rng, err := pgf.PosRange(start, end)
	if err != nil {
		return protocol.Range{}, nil, err
	}

	var b strings.Builder
	if value != "" {
		b.WriteString(value)
	}
	if r != 0 {
		runeName := runenames.Name(r)
		if len(runeName) > 0 && runeName[0] == '<' {
			// Check if the rune looks like an HTML tag. If so, trim the surrounding <>
			// characters to work around https://github.com/microsoft/vscode/issues/124042.
			runeName = strings.TrimRight(runeName[1:], ">")
		}
		if b.Len() > 0 {
			b.WriteString(", ")
		}
		if strconv.IsPrint(r) {
			fmt.Fprintf(&b, "'%c', ", r)
		}
		fmt.Fprintf(&b, "U+%04X, %s", r, runeName)
	}
	hover := b.String()
	return rng, &HoverJSON{
		Synopsis:          hover,
		FullDocumentation: hover,
	}, nil
}

// hoverEmbed computes hover information for a filepath.Match pattern.
// Assumes that the pattern is relative to the location of fh.
func hoverEmbed(fh FileHandle, rng protocol.Range, pattern string) (protocol.Range, *HoverJSON, error) {
	s := &strings.Builder{}

	dir := filepath.Dir(fh.URI().Filename())
	var matches []string
	err := filepath.WalkDir(dir, func(abs string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		rel, err := filepath.Rel(dir, abs)
		if err != nil {
			return err
		}
		ok, err := filepath.Match(pattern, rel)
		if err != nil {
			return err
		}
		if ok && !d.IsDir() {
			matches = append(matches, rel)
		}
		return nil
	})
	if err != nil {
		return protocol.Range{}, nil, err
	}

	for _, m := range matches {
		// TODO: Renders each file as separate markdown paragraphs.
		// If forcing (a single) newline is possible it might be more clear.
		fmt.Fprintf(s, "%s\n\n", m)
	}

	json := &HoverJSON{
		Signature:         fmt.Sprintf("Embedding %q", pattern),
		Synopsis:          s.String(),
		FullDocumentation: s.String(),
	}
	return rng, json, nil
}

// inferredSignatureString is a wrapper around the types.ObjectString function
// that adds more information to inferred signatures. It will return an empty string
// if the passed types.Object is not a signature.
func inferredSignatureString(obj types.Object, qf types.Qualifier, inferred *types.Signature) string {
	// If the signature type was inferred, prefer the inferred signature with a
	// comment showing the generic signature.
	if sig, _ := obj.Type().(*types.Signature); sig != nil && typeparams.ForSignature(sig).Len() > 0 && inferred != nil {
		obj2 := types.NewFunc(obj.Pos(), obj.Pkg(), obj.Name(), inferred)
		str := types.ObjectString(obj2, qf)
		// Try to avoid overly long lines.
		if len(str) > 60 {
			str += "\n"
		} else {
			str += " "
		}
		str += "// " + types.TypeString(sig, qf)
		return str
	}
	return ""
}

// objectString is a wrapper around the types.ObjectString function.
// It handles adding more information to the object string.
// If spec is non-nil, it may be used to format additional declaration
// syntax, and file must be the token.File describing its positions.
func objectString(obj types.Object, qf types.Qualifier, declPos token.Pos, file *token.File, spec ast.Spec) string {
	str := types.ObjectString(obj, qf)

	switch obj := obj.(type) {
	case *types.Const:
		var (
			declaration = obj.Val().String() // default formatted declaration
			comment     = ""                 // if non-empty, a clarifying comment
		)

		// Try to use the original declaration.
		switch obj.Val().Kind() {
		case constant.String:
			// Usually the original declaration of a string doesn't carry much information.
			// Also strings can be very long. So, just use the constant's value.

		default:
			if spec, _ := spec.(*ast.ValueSpec); spec != nil {
				for i, name := range spec.Names {
					if declPos == name.Pos() {
						if i < len(spec.Values) {
							originalDeclaration := FormatNodeFile(file, spec.Values[i])
							if originalDeclaration != declaration {
								comment = declaration
								declaration = originalDeclaration
							}
						}
						break
					}
				}
			}
		}

		// Special formatting cases.
		switch typ := obj.Type().(type) {
		case *types.Named:
			// Try to add a formatted duration as an inline comment.
			pkg := typ.Obj().Pkg()
			if pkg.Path() == "time" && typ.Obj().Name() == "Duration" {
				if d, ok := constant.Int64Val(obj.Val()); ok {
					comment = time.Duration(d).String()
				}
			}
		}
		if comment == declaration {
			comment = ""
		}

		str += " = " + declaration
		if comment != "" {
			str += " // " + comment
		}
	}
	return str
}

// HoverDocForObject returns the best doc comment for obj (for which
// fset provides file/line information).
//
// TODO(rfindley): there appears to be zero(!) tests for this functionality.
func HoverDocForObject(ctx context.Context, snapshot Snapshot, fset *token.FileSet, obj types.Object) (*ast.CommentGroup, error) {
	if _, isTypeName := obj.(*types.TypeName); isTypeName {
		if _, isTypeParam := obj.Type().(*typeparams.TypeParam); isTypeParam {
			return nil, nil
		}
	}

	pgf, pos, err := parseFull(ctx, snapshot, fset, obj.Pos())
	if err != nil {
		return nil, fmt.Errorf("re-parsing: %v", err)
	}

	decl, spec, field := findDeclInfo([]*ast.File{pgf.File}, pos)
	return chooseDocComment(decl, spec, field), nil
}

func chooseDocComment(decl ast.Decl, spec ast.Spec, field *ast.Field) *ast.CommentGroup {
	if field != nil {
		if field.Doc != nil {
			return field.Doc
		}
		if field.Comment != nil {
			return field.Comment
		}
		return nil
	}
	switch decl := decl.(type) {
	case *ast.FuncDecl:
		return decl.Doc
	case *ast.GenDecl:
		switch spec := spec.(type) {
		case *ast.ValueSpec:
			if spec.Doc != nil {
				return spec.Doc
			}
			if decl.Doc != nil {
				return decl.Doc
			}
			return spec.Comment
		case *ast.TypeSpec:
			if spec.Doc != nil {
				return spec.Doc
			}
			if decl.Doc != nil {
				return decl.Doc
			}
			return spec.Comment
		}
	}
	return nil
}

// parseFull fully parses the file corresponding to position pos (for
// which fset provides file/line information).
//
// It returns the resulting ParsedGoFile as well as new pos contained in the
// parsed file.
func parseFull(ctx context.Context, snapshot Snapshot, fset *token.FileSet, pos token.Pos) (*ParsedGoFile, token.Pos, error) {
	f := fset.File(pos)
	if f == nil {
		return nil, 0, bug.Errorf("internal error: no file for position %d", pos)
	}

	uri := span.URIFromPath(f.Name())
	fh, err := snapshot.ReadFile(ctx, uri)
	if err != nil {
		return nil, 0, err
	}

	pgf, err := snapshot.ParseGo(ctx, fh, ParseFull)
	if err != nil {
		return nil, 0, err
	}

	offset, err := safetoken.Offset(f, pos)
	if err != nil {
		return nil, 0, bug.Errorf("offset out of bounds in %q", uri)
	}

	fullPos, err := safetoken.Pos(pgf.Tok, offset)
	if err != nil {
		return nil, 0, err
	}

	return pgf, fullPos, nil
}

func formatHover(h *HoverJSON, options *Options) (string, error) {
	signature := formatSignature(h, options)

	switch options.HoverKind {
	case SingleLine:
		return h.SingleLine, nil
	case NoDocumentation:
		return signature, nil
	case Structured:
		b, err := json.Marshal(h)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}

	link := formatLink(h, options)
	doc := formatDoc(h, options)

	var b strings.Builder
	parts := []string{signature, doc, link}
	for i, el := range parts {
		if el != "" {
			b.WriteString(el)

			// If any elements of the remainder of the list are non-empty,
			// write an extra newline.
			if anyNonEmpty(parts[i+1:]) {
				if options.PreferredContentFormat == protocol.Markdown {
					b.WriteString("\n\n")
				} else {
					b.WriteRune('\n')
				}
			}
		}
	}
	return b.String(), nil
}

func formatSignature(h *HoverJSON, options *Options) string {
	signature := h.Signature
	if signature != "" && options.PreferredContentFormat == protocol.Markdown {
		signature = fmt.Sprintf("```go\n%s\n```", signature)
	}
	return signature
}

func formatLink(h *HoverJSON, options *Options) string {
	if !options.LinksInHover || options.LinkTarget == "" || h.LinkPath == "" {
		return ""
	}
	plainLink := BuildLink(options.LinkTarget, h.LinkPath, h.LinkAnchor)
	switch options.PreferredContentFormat {
	case protocol.Markdown:
		return fmt.Sprintf("[`%s` on %s](%s)", h.SymbolName, options.LinkTarget, plainLink)
	case protocol.PlainText:
		return ""
	default:
		return plainLink
	}
}

// BuildLink constructs a URL with the given target, path, and anchor.
func BuildLink(target, path, anchor string) string {
	link := fmt.Sprintf("https://%s/%s", target, path)
	if anchor == "" {
		return link
	}
	return link + "#" + anchor
}

func formatDoc(h *HoverJSON, options *Options) string {
	var doc string
	switch options.HoverKind {
	case SynopsisDocumentation:
		doc = h.Synopsis
	case FullDocumentation:
		doc = h.FullDocumentation
	}
	if options.PreferredContentFormat == protocol.Markdown {
		return CommentToMarkdown(doc, options)
	}
	return doc
}

func anyNonEmpty(x []string) bool {
	for _, el := range x {
		if el != "" {
			return true
		}
	}
	return false
}

// findDeclInfo returns the syntax nodes involved in the declaration of the
// types.Object with position pos, searching the given list of file syntax
// trees.
//
// Pos may be the position of the name-defining identifier in a FuncDecl,
// ValueSpec, TypeSpec, Field, or as a special case the position of
// Ellipsis.Elt in an ellipsis field.
//
// If found, the resulting decl, spec, and field will be the inner-most
// instance of each node type surrounding pos.
//
// If field is non-nil, pos is the position of a field Var. If field is nil and
// spec is non-nil, pos is the position of a Var, Const, or TypeName object. If
// both field and spec are nil and decl is non-nil, pos is the position of a
// Func object.
//
// It returns a nil decl if no object-defining node is found at pos.
//
// TODO(rfindley): this function has tricky semantics, and may be worth unit
// testing and/or refactoring.
func findDeclInfo(files []*ast.File, pos token.Pos) (decl ast.Decl, spec ast.Spec, field *ast.Field) {
	// panic(found{}) breaks off the traversal and
	// causes the function to return normally.
	type found struct{}
	defer func() {
		switch x := recover().(type) {
		case nil:
		case found:
		default:
			panic(x)
		}
	}()

	// Visit the files in search of the node at pos.
	stack := make([]ast.Node, 0, 20)
	// Allocate the closure once, outside the loop.
	f := func(n ast.Node) bool {
		if n != nil {
			stack = append(stack, n) // push
		} else {
			stack = stack[:len(stack)-1] // pop
			return false
		}

		// Skip subtrees (incl. files) that don't contain the search point.
		if !(n.Pos() <= pos && pos < n.End()) {
			return false
		}

		switch n := n.(type) {
		case *ast.Field:
			findEnclosingDeclAndSpec := func() {
				for i := len(stack) - 1; i >= 0; i-- {
					switch n := stack[i].(type) {
					case ast.Spec:
						spec = n
					case ast.Decl:
						decl = n
						return
					}
				}
			}

			// Check each field name since you can have
			// multiple names for the same type expression.
			for _, id := range n.Names {
				if id.Pos() == pos {
					field = n
					findEnclosingDeclAndSpec()
					panic(found{})
				}
			}

			// Check *ast.Field itself. This handles embedded
			// fields which have no associated *ast.Ident name.
			if n.Pos() == pos {
				field = n
				findEnclosingDeclAndSpec()
				panic(found{})
			}

			// Also check "X" in "...X". This makes it easy to format variadic
			// signature params properly.
			//
			// TODO(rfindley): I don't understand this comment. How does finding the
			// field in this case make it easier to format variadic signature params?
			if ell, ok := n.Type.(*ast.Ellipsis); ok && ell.Elt != nil && ell.Elt.Pos() == pos {
				field = n
				findEnclosingDeclAndSpec()
				panic(found{})
			}

		case *ast.FuncDecl:
			if n.Name.Pos() == pos {
				decl = n
				panic(found{})
			}

		case *ast.GenDecl:
			for _, s := range n.Specs {
				switch s := s.(type) {
				case *ast.TypeSpec:
					if s.Name.Pos() == pos {
						decl = n
						spec = s
						panic(found{})
					}
				case *ast.ValueSpec:
					for _, id := range s.Names {
						if id.Pos() == pos {
							decl = n
							spec = s
							panic(found{})
						}
					}
				}
			}
		}
		return true
	}
	for _, file := range files {
		ast.Inspect(file, f)
	}

	return nil, nil, nil
}
