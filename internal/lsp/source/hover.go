// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/doc"
	"go/format"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

type HoverInformation struct {
	// Signature is the symbol's signature.
	Signature string `json:"signature"`

	// SingleLine is a single line describing the symbol.
	// This is recommended only for use in clients that show a single line for hover.
	SingleLine string `json:"singleLine"`

	// Synopsis is a single sentence synopsis of the symbol's documentation.
	Synopsis string `json:"synopsis"`

	// FullDocumentation is the symbol's full documentation.
	FullDocumentation string `json:"fullDocumentation"`

	// Link is the pkg.go.dev anchor for the given symbol.
	// For example, "go/ast#Node".
	Link string `json:"link"`

	// SymbolName is the types.Object.Name for the given symbol.
	SymbolName string

	source  interface{}
	comment *ast.CommentGroup
}

func Hover(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) (*protocol.Hover, error) {
	ident, err := Identifier(ctx, snapshot, fh, position)
	if err != nil {
		return nil, nil
	}
	h, err := ident.Hover(ctx)
	if err != nil {
		return nil, err
	}
	rng, err := ident.Range()
	if err != nil {
		return nil, err
	}
	isPrivate := h.Link != "" && snapshot.View().IsGoPrivatePath(h.Link)
	hover, err := FormatHover(h, snapshot.View().Options(), isPrivate)
	if err != nil {
		return nil, err
	}
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  snapshot.View().Options().PreferredContentFormat,
			Value: hover,
		},
		Range: rng,
	}, nil
}

func (i *IdentifierInfo) Hover(ctx context.Context) (*HoverInformation, error) {
	ctx, done := event.Start(ctx, "source.Hover")
	defer done()

	fset := i.Snapshot.View().Session().Cache().FileSet()
	h, err := hover(ctx, fset, i.pkg, i.Declaration)
	if err != nil {
		return nil, err
	}
	// Determine the symbol's signature.
	switch x := h.source.(type) {
	case ast.Node:
		var b strings.Builder
		if err := format.Node(&b, fset, x); err != nil {
			return nil, err
		}
		h.Signature = b.String()
	case types.Object:
		h.Signature = objectString(x, i.qf)
	}
	if obj := i.Declaration.obj; obj != nil {
		h.SingleLine = objectString(obj, i.qf)
	}
	h.Link, h.SymbolName = i.linkAndSymbolName()
	if h.comment != nil {
		h.FullDocumentation = h.comment.Text()
		h.Synopsis = doc.Synopsis(h.FullDocumentation)
	}
	return h, nil
}

func (i *IdentifierInfo) linkAndSymbolName() (string, string) {
	obj := i.Declaration.obj
	if obj == nil {
		return "", ""
	}
	switch obj := obj.(type) {
	case *types.PkgName:
		path := obj.Imported().Path()
		if mod, version, ok := moduleAtVersion(path, i); ok {
			path = strings.Replace(path, mod, mod+"@"+version, 1)
		}
		return path, obj.Name()
	case *types.Builtin:
		return fmt.Sprintf("builtin#%s", obj.Name()), obj.Name()
	}
	// Check if the identifier is test-only (and is therefore not part of a
	// package's API). This is true if the request originated in a test package,
	// and if the declaration is also found in the same test package.
	if i.pkg != nil && obj.Pkg() != nil && i.pkg.ForTest() != "" {
		if _, pkg, _ := FindFileInPackage(i.pkg, i.Declaration.MappedRange[0].URI()); i.pkg == pkg {
			return "", ""
		}
	}
	// Don't return links for other unexported types.
	if !obj.Exported() {
		return "", ""
	}
	var rTypeName string
	switch obj := obj.(type) {
	case *types.Var:
		if obj.IsField() {
			// If the object is a field, and we have an associated selector
			// composite literal, or struct, we can determine the link.
			switch typ := i.enclosing.(type) {
			case *types.Named:
				rTypeName = typ.Obj().Name()
			}
		}
	case *types.Func:
		typ, ok := obj.Type().(*types.Signature)
		if !ok {
			return "", ""
		}
		if r := typ.Recv(); r != nil {
			switch rtyp := deref(r.Type()).(type) {
			case *types.Struct:
				rTypeName = r.Name()
			case *types.Named:
				if named, ok := i.enclosing.(*types.Named); ok {
					rTypeName = named.Obj().Name()
				} else if !rtyp.Obj().Exported() {
					return "", ""
				} else {
					rTypeName = rtyp.Obj().Name()
				}
			}
		}
	}
	path := obj.Pkg().Path()
	if mod, version, ok := moduleAtVersion(path, i); ok {
		path = strings.Replace(path, mod, mod+"@"+version, 1)
	}
	if rTypeName != "" {
		link := fmt.Sprintf("%s#%s.%s", path, rTypeName, obj.Name())
		symbol := fmt.Sprintf("(%s.%s).%s", obj.Pkg().Name(), rTypeName, obj.Name())
		return link, symbol
	}
	// For most cases, the link is "package/path#symbol".
	return fmt.Sprintf("%s#%s", path, obj.Name()), fmt.Sprintf("%s.%s", obj.Pkg().Name(), obj.Name())
}

func moduleAtVersion(path string, i *IdentifierInfo) (string, string, bool) {
	if strings.ToLower(i.Snapshot.View().Options().LinkTarget) != "pkg.go.dev" {
		return "", "", false
	}
	impPkg, err := i.pkg.GetImport(path)
	if err != nil {
		return "", "", false
	}
	if impPkg.Module() == nil {
		return "", "", false
	}
	version, modpath := impPkg.Module().Version, impPkg.Module().Path
	if modpath == "" || version == "" {
		return "", "", false
	}
	return modpath, version, true
}

// objectString is a wrapper around the types.ObjectString function.
// It handles adding more information to the object string.
func objectString(obj types.Object, qf types.Qualifier) string {
	str := types.ObjectString(obj, qf)
	switch obj := obj.(type) {
	case *types.Const:
		str = fmt.Sprintf("%s = %s", str, obj.Val())
	}
	return str
}

func hover(ctx context.Context, fset *token.FileSet, pkg Package, d Declaration) (*HoverInformation, error) {
	_, done := event.Start(ctx, "source.hover")
	defer done()

	obj := d.obj
	switch node := d.node.(type) {
	case *ast.Ident:
		// The package declaration.
		for _, f := range pkg.GetSyntax() {
			if f.Name == node {
				return &HoverInformation{comment: f.Doc}, nil
			}
		}
	case *ast.ImportSpec:
		// Try to find the package documentation for an imported package.
		if pkgName, ok := obj.(*types.PkgName); ok {
			imp, err := pkg.GetImport(pkgName.Imported().Path())
			if err != nil {
				return nil, err
			}
			// Assume that only one file will contain package documentation,
			// so pick the first file that has a doc comment.
			var doc *ast.CommentGroup
			for _, file := range imp.GetSyntax() {
				if file.Doc != nil {
					return &HoverInformation{source: obj, comment: doc}, nil
				}
			}
		}
		return &HoverInformation{source: node}, nil
	case *ast.GenDecl:
		switch obj := obj.(type) {
		case *types.TypeName, *types.Var, *types.Const, *types.Func:
			return formatGenDecl(node, obj, obj.Type())
		}
	case *ast.TypeSpec:
		if obj.Parent() == types.Universe {
			if obj.Name() == "error" {
				return &HoverInformation{source: node}, nil
			}
			return &HoverInformation{source: node.Name}, nil // comments not needed for builtins
		}
	case *ast.FuncDecl:
		switch obj.(type) {
		case *types.Func:
			return &HoverInformation{source: obj, comment: node.Doc}, nil
		case *types.Builtin:
			return &HoverInformation{source: node.Type, comment: node.Doc}, nil
		}
	}
	return &HoverInformation{source: obj}, nil
}

func formatGenDecl(node *ast.GenDecl, obj types.Object, typ types.Type) (*HoverInformation, error) {
	if _, ok := typ.(*types.Named); ok {
		switch typ.Underlying().(type) {
		case *types.Interface, *types.Struct:
			return formatGenDecl(node, obj, typ.Underlying())
		}
	}
	var spec ast.Spec
	for _, s := range node.Specs {
		if s.Pos() <= obj.Pos() && obj.Pos() <= s.End() {
			spec = s
			break
		}
	}
	if spec == nil {
		return nil, errors.Errorf("no spec for node %v at position %v", node, obj.Pos())
	}
	// If we have a field or method.
	switch obj.(type) {
	case *types.Var, *types.Const, *types.Func:
		return formatVar(spec, obj, node), nil
	}
	// Handle types.
	switch spec := spec.(type) {
	case *ast.TypeSpec:
		if len(node.Specs) > 1 {
			// If multiple types are declared in the same block.
			return &HoverInformation{source: spec.Type, comment: spec.Doc}, nil
		} else {
			return &HoverInformation{source: spec, comment: node.Doc}, nil
		}
	case *ast.ValueSpec:
		return &HoverInformation{source: spec, comment: spec.Doc}, nil
	case *ast.ImportSpec:
		return &HoverInformation{source: spec, comment: spec.Doc}, nil
	}
	return nil, errors.Errorf("unable to format spec %v (%T)", spec, spec)
}

func formatVar(node ast.Spec, obj types.Object, decl *ast.GenDecl) *HoverInformation {
	var fieldList *ast.FieldList
	switch spec := node.(type) {
	case *ast.TypeSpec:
		switch t := spec.Type.(type) {
		case *ast.StructType:
			fieldList = t.Fields
		case *ast.InterfaceType:
			fieldList = t.Methods
		}
	case *ast.ValueSpec:
		comment := spec.Doc
		if comment == nil {
			comment = decl.Doc
		}
		if comment == nil {
			comment = spec.Comment
		}
		return &HoverInformation{source: obj, comment: comment}
	}
	// If we have a struct or interface declaration,
	// we need to match the object to the corresponding field or method.
	if fieldList != nil {
		for i := 0; i < len(fieldList.List); i++ {
			field := fieldList.List[i]
			if field.Pos() <= obj.Pos() && obj.Pos() <= field.End() {
				if field.Doc.Text() != "" {
					return &HoverInformation{source: obj, comment: field.Doc}
				}
				return &HoverInformation{source: obj, comment: field.Comment}
			}
		}
	}
	return &HoverInformation{source: obj, comment: decl.Doc}
}

func FormatHover(h *HoverInformation, options Options, isPrivate bool) (string, error) {
	signature := h.Signature
	if signature != "" && options.PreferredContentFormat == protocol.Markdown {
		signature = fmt.Sprintf("```go\n%s\n```", signature)
	}

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
	var link string
	if !isPrivate {
		link = formatLink(h, options)
	}
	switch options.HoverKind {
	case SynopsisDocumentation:
		doc := formatDoc(h.Synopsis, options)
		return formatHover(options, doc, link, signature), nil
	case FullDocumentation:
		doc := formatDoc(h.FullDocumentation, options)
		return formatHover(options, signature, link, doc), nil
	}
	return "", errors.Errorf("no hover for %v", h.source)
}

func formatLink(h *HoverInformation, options Options) string {
	if options.LinkTarget == "" || h.Link == "" {
		return ""
	}
	plainLink := fmt.Sprintf("https://%s/%s", options.LinkTarget, h.Link)
	switch options.PreferredContentFormat {
	case protocol.Markdown:
		return fmt.Sprintf("[`%s` on %s](%s)", h.SymbolName, options.LinkTarget, plainLink)
	case protocol.PlainText:
		return ""
	default:
		return plainLink
	}
}
func formatDoc(doc string, options Options) string {
	if options.PreferredContentFormat == protocol.Markdown {
		return CommentToMarkdown(doc)
	}
	return doc
}

func formatHover(options Options, x ...string) string {
	var b strings.Builder
	for i, el := range x {
		if el != "" {
			b.WriteString(el)

			// Don't write out final newline.
			if i == len(x) {
				continue
			}
			// If any elements of the remainder of the list are non-empty,
			// write a newline.
			if anyNonEmpty(x[i+1:]) {
				if options.PreferredContentFormat == protocol.Markdown {
					b.WriteString("\n\n")
				} else {
					b.WriteRune('\n')
				}
			}
		}
	}
	return b.String()
}

func anyNonEmpty(x []string) bool {
	for _, el := range x {
		if el != "" {
			return true
		}
	}
	return false
}
