// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/constant"
	"go/doc"
	"go/format"
	"go/token"
	"go/types"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"golang.org/x/text/unicode/runenames"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/typeparams"
)

// HoverContext contains context extracted from the syntax and type information
// of a given node, for use in various summaries (hover, autocomplete,
// signature help).
type HoverContext struct {
	// signatureSource is the object or node use to derive the hover signature.
	//
	// It may also hold a precomputed string.
	// TODO(rfindley): pre-compute all signatures to avoid this indirection.
	signatureSource interface{}

	// comment is the most relevant comment group associated with the hovered object.
	Comment *ast.CommentGroup
}

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

	// SymbolName is the types.Object.Name for the given symbol.
	SymbolName string `json:"symbolName"`

	// LinkPath is the pkg.go.dev link for the given symbol.
	// For example, the "go/ast" part of "pkg.go.dev/go/ast#Node".
	LinkPath string `json:"linkPath"`

	// LinkAnchor is the pkg.go.dev link anchor for the given symbol.
	// For example, the "Node" part of "pkg.go.dev/go/ast#Node".
	LinkAnchor string `json:"linkAnchor"`
}

func Hover(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) (*protocol.Hover, error) {
	ident, err := Identifier(ctx, snapshot, fh, position)
	if err != nil {
		if hover, innerErr := hoverRune(ctx, snapshot, fh, position); innerErr == nil {
			return hover, nil
		}
		return nil, nil
	}
	h, err := HoverIdentifier(ctx, ident)
	if err != nil {
		return nil, err
	}
	rng, err := ident.Range()
	if err != nil {
		return nil, err
	}
	hover, err := FormatHover(h, snapshot.View().Options())
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

func hoverRune(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) (*protocol.Hover, error) {
	ctx, done := event.Start(ctx, "source.hoverRune")
	defer done()

	r, mrng, err := findRune(ctx, snapshot, fh, position)
	if err != nil {
		return nil, err
	}
	rng, err := mrng.Range()
	if err != nil {
		return nil, err
	}

	var desc string
	runeName := runenames.Name(r)
	if len(runeName) > 0 && runeName[0] == '<' {
		// Check if the rune looks like an HTML tag. If so, trim the surrounding <>
		// characters to work around https://github.com/microsoft/vscode/issues/124042.
		runeName = strings.TrimRight(runeName[1:], ">")
	}
	if strconv.IsPrint(r) {
		desc = fmt.Sprintf("'%s', U+%04X, %s", string(r), uint32(r), runeName)
	} else {
		desc = fmt.Sprintf("U+%04X, %s", uint32(r), runeName)
	}
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  snapshot.View().Options().PreferredContentFormat,
			Value: desc,
		},
		Range: rng,
	}, nil
}

// ErrNoRuneFound is the error returned when no rune is found at a particular position.
var ErrNoRuneFound = errors.New("no rune found")

// findRune returns rune information for a position in a file.
func findRune(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) (rune, MappedRange, error) {
	// TODO(adonovan): opt: avoid loading type-checked package; only parsing is needed.
	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return 0, MappedRange{}, err
	}
	pos, err := pgf.Mapper.Pos(position)
	if err != nil {
		return 0, MappedRange{}, err
	}

	// Find the basic literal enclosing the given position, if there is one.
	var lit *ast.BasicLit
	var found bool
	ast.Inspect(pgf.File, func(n ast.Node) bool {
		if found {
			return false
		}
		if n, ok := n.(*ast.BasicLit); ok && pos >= n.Pos() && pos <= n.End() {
			lit = n
			found = true
		}
		return !found
	})
	if !found {
		return 0, MappedRange{}, ErrNoRuneFound
	}

	var r rune
	var start, end token.Pos
	switch lit.Kind {
	case token.CHAR:
		s, err := strconv.Unquote(lit.Value)
		if err != nil {
			// If the conversion fails, it's because of an invalid syntax, therefore
			// there is no rune to be found.
			return 0, MappedRange{}, ErrNoRuneFound
		}
		r, _ = utf8.DecodeRuneInString(s)
		if r == utf8.RuneError {
			return 0, MappedRange{}, fmt.Errorf("rune error")
		}
		start, end = lit.Pos(), lit.End()
	case token.INT:
		// It's an integer, scan only if it is a hex litteral whose bitsize in
		// ranging from 8 to 32.
		if !(strings.HasPrefix(lit.Value, "0x") && len(lit.Value[2:]) >= 2 && len(lit.Value[2:]) <= 8) {
			return 0, MappedRange{}, ErrNoRuneFound
		}
		v, err := strconv.ParseUint(lit.Value[2:], 16, 32)
		if err != nil {
			return 0, MappedRange{}, err
		}
		r = rune(v)
		if r == utf8.RuneError {
			return 0, MappedRange{}, fmt.Errorf("rune error")
		}
		start, end = lit.Pos(), lit.End()
	case token.STRING:
		// It's a string, scan only if it contains a unicode escape sequence under or before the
		// current cursor position.
		var found bool
		litOffset, err := safetoken.Offset(pgf.Tok, lit.Pos())
		if err != nil {
			return 0, MappedRange{}, err
		}
		offset, err := safetoken.Offset(pgf.Tok, pos)
		if err != nil {
			return 0, MappedRange{}, err
		}
		for i := offset - litOffset; i > 0; i-- {
			// Start at the cursor position and search backward for the beginning of a rune escape sequence.
			rr, _ := utf8.DecodeRuneInString(lit.Value[i:])
			if rr == utf8.RuneError {
				return 0, MappedRange{}, fmt.Errorf("rune error")
			}
			if rr == '\\' {
				// Got the beginning, decode it.
				var tail string
				r, _, tail, err = strconv.UnquoteChar(lit.Value[i:], '"')
				if err != nil {
					// If the conversion fails, it's because of an invalid syntax, therefore is no rune to be found.
					return 0, MappedRange{}, ErrNoRuneFound
				}
				// Only the rune escape sequence part of the string has to be highlighted, recompute the range.
				runeLen := len(lit.Value) - (int(i) + len(tail))
				start = token.Pos(int(lit.Pos()) + int(i))
				end = token.Pos(int(start) + runeLen)
				found = true
				break
			}
		}
		if !found {
			// No escape sequence found
			return 0, MappedRange{}, ErrNoRuneFound
		}
	default:
		return 0, MappedRange{}, ErrNoRuneFound
	}

	mappedRange, err := posToMappedRange(pkg, start, end)
	if err != nil {
		return 0, MappedRange{}, err
	}
	return r, mappedRange, nil
}

func HoverIdentifier(ctx context.Context, i *IdentifierInfo) (*HoverJSON, error) {
	ctx, done := event.Start(ctx, "source.Hover")
	defer done()

	hoverCtx, err := FindHoverContext(ctx, i.Snapshot, i.pkg, i.Declaration.obj, i.Declaration.node, i.Declaration.fullDecl)
	if err != nil {
		return nil, err
	}

	h := &HoverJSON{
		FullDocumentation: hoverCtx.Comment.Text(),
		Synopsis:          doc.Synopsis(hoverCtx.Comment.Text()),
	}

	fset := i.pkg.FileSet()
	// Determine the symbol's signature.
	switch x := hoverCtx.signatureSource.(type) {
	case string:
		h.Signature = x // a pre-computed signature

	case *ast.TypeSpec:
		x2 := *x
		// Don't duplicate comments when formatting type specs.
		x2.Doc = nil
		x2.Comment = nil
		var b strings.Builder
		b.WriteString("type ")
		if err := format.Node(&b, fset, &x2); err != nil {
			return nil, err
		}

		// Display the declared methods accessible from the identifier.
		//
		// (The format.Node call above displays any struct fields, public
		// or private, in syntactic form. We choose not to recursively
		// enumerate any fields and methods promoted from them.)
		obj := i.Type.Object
		if obj != nil && !types.IsInterface(obj.Type()) {
			sep := "\n\n"
			for _, m := range typeutil.IntuitiveMethodSet(obj.Type(), nil) {
				if (m.Obj().Exported() || m.Obj().Pkg() == i.pkg.GetTypes()) && len(m.Index()) == 1 {
					b.WriteString(sep)
					sep = "\n"
					b.WriteString(objectString(m.Obj(), i.qf, nil))
				}
			}
		}

		h.Signature = b.String()

	case ast.Node:
		var b strings.Builder
		if err := format.Node(&b, fset, x); err != nil {
			return nil, err
		}
		h.Signature = b.String()

		// Check if the variable is an integer whose value we can present in a more
		// user-friendly way, i.e. `var hex = 0xe34e` becomes `var hex = 58190`
		if spec, ok := x.(*ast.ValueSpec); ok && len(spec.Values) > 0 {
			if lit, ok := spec.Values[0].(*ast.BasicLit); ok && len(spec.Names) > 0 {
				val := constant.MakeFromLiteral(types.ExprString(lit), lit.Kind, 0)
				h.Signature = fmt.Sprintf("var %s = %s", spec.Names[0], val)
			}
		}

	case types.Object:
		// If the variable is implicitly declared in a type switch, we need to
		// manually generate its object string.
		if typ := i.Declaration.typeSwitchImplicit; typ != nil {
			if v, ok := x.(*types.Var); ok {
				h.Signature = fmt.Sprintf("var %s %s", v.Name(), types.TypeString(typ, i.qf))
				break
			}
		}
		h.Signature = objectString(x, i.qf, i.Inferred)
	}
	if obj := i.Declaration.obj; obj != nil {
		h.SingleLine = objectString(obj, i.qf, nil)
	}
	obj := i.Declaration.obj
	if obj == nil {
		return h, nil
	}

	// Check if the identifier is test-only (and is therefore not part of a
	// package's API). This is true if the request originated in a test package,
	// and if the declaration is also found in the same test package.
	if i.pkg != nil && obj.Pkg() != nil && i.pkg.ForTest() != "" {
		if _, err := i.pkg.File(i.Declaration.MappedRange[0].URI()); err == nil {
			return h, nil
		}
	}

	h.SymbolName, h.LinkPath, h.LinkAnchor = linkData(obj, i.enclosing)

	// See golang/go#36998: don't link to modules matching GOPRIVATE.
	//
	// The path returned by linkData is a package path.
	if i.Snapshot.View().IsGoPrivatePath(h.LinkPath) {
		h.LinkPath = ""
	} else if mod, version, ok := moduleAtVersion(h.LinkPath, i); ok {
		h.LinkPath = strings.Replace(h.LinkPath, mod, mod+"@"+version, 1)
	}

	return h, nil
}

// linkData returns the name, package path, and anchor to use in building links
// to obj.
//
// If obj is not visible in documentation, the returned name will be empty.
func linkData(obj types.Object, enclosing *types.TypeName) (name, packagePath, anchor string) {
	// Package names simply link to the package.
	if obj, ok := obj.(*types.PkgName); ok {
		return obj.Name(), obj.Imported().Path(), ""
	}

	// Builtins link to the special builtin package.
	if obj.Parent() == types.Universe {
		return obj.Name(), "builtin", obj.Name()
	}

	// In all other cases, the object must be exported.
	if !obj.Exported() {
		return "", "", ""
	}

	var recv types.Object // If non-nil, the field or method receiver base.

	switch obj := obj.(type) {
	case *types.Var:
		// If the object is a field, and we have an associated selector
		// composite literal, or struct, we can determine the link.
		if obj.IsField() && enclosing != nil {
			recv = enclosing
		}
	case *types.Func:
		typ, ok := obj.Type().(*types.Signature)
		if !ok {
			// Note: this should never happen. go/types guarantees that the type of
			// *Funcs are Signatures.
			//
			// TODO(rfindley): given a 'debug' mode, we should panic here.
			return "", "", ""
		}
		if r := typ.Recv(); r != nil {
			if rtyp, _ := Deref(r.Type()).(*types.Named); rtyp != nil {
				// If we have an unexported type, see if the enclosing type is
				// exported (we may have an interface or struct we can link
				// to). If not, don't show any link.
				if !rtyp.Obj().Exported() {
					if enclosing != nil {
						recv = enclosing
					} else {
						return "", "", ""
					}
				} else {
					recv = rtyp.Obj()
				}
			}
		}
	}

	if recv != nil && !recv.Exported() {
		return "", "", ""
	}

	// Either the object or its receiver must be in the package scope.
	scopeObj := obj
	if recv != nil {
		scopeObj = recv
	}
	if scopeObj.Pkg() == nil || scopeObj.Pkg().Scope().Lookup(scopeObj.Name()) != scopeObj {
		return "", "", ""
	}

	// golang/go#52211: somehow we get here with a nil obj.Pkg
	if obj.Pkg() == nil {
		bug.Report("object with nil pkg", bug.Data{
			"name": obj.Name(),
			"type": fmt.Sprintf("%T", obj),
		})
		return "", "", ""
	}

	packagePath = obj.Pkg().Path()
	if recv != nil {
		anchor = fmt.Sprintf("%s.%s", recv.Name(), obj.Name())
		name = fmt.Sprintf("(%s.%s).%s", obj.Pkg().Name(), recv.Name(), obj.Name())
	} else {
		// For most cases, the link is "package/path#symbol".
		anchor = obj.Name()
		name = fmt.Sprintf("%s.%s", obj.Pkg().Name(), obj.Name())
	}
	return name, packagePath, anchor
}

func moduleAtVersion(path string, i *IdentifierInfo) (string, string, bool) {
	// TODO(rfindley): moduleAtVersion should not be responsible for deciding
	// whether or not the link target supports module version links.
	if strings.ToLower(i.Snapshot.View().Options().LinkTarget) != "pkg.go.dev" {
		return "", "", false
	}
	impPkg, err := i.pkg.DirectDep(PackagePath(path))
	if err != nil {
		return "", "", false
	}
	if impPkg.Version() == nil {
		return "", "", false
	}
	version, modpath := impPkg.Version().Version, impPkg.Version().Path
	if modpath == "" || version == "" {
		return "", "", false
	}
	return modpath, version, true
}

// objectString is a wrapper around the types.ObjectString function.
// It handles adding more information to the object string.
func objectString(obj types.Object, qf types.Qualifier, inferred *types.Signature) string {
	// If the signature type was inferred, prefer the preferred signature with a
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
	str := types.ObjectString(obj, qf)
	switch obj := obj.(type) {
	case *types.Const:
		str = fmt.Sprintf("%s = %s", str, obj.Val())

		// Try to add a formatted duration as an inline comment
		typ, ok := obj.Type().(*types.Named)
		if !ok {
			break
		}
		pkg := typ.Obj().Pkg()
		if pkg.Path() == "time" && typ.Obj().Name() == "Duration" {
			if d, ok := constant.Int64Val(obj.Val()); ok {
				str += " // " + time.Duration(d).String()
			}
		}
	}
	return str
}

// FindHoverContext returns a HoverContext struct for an AST node and its
// declaration object. node should be the actual node used in type checking,
// while fullNode could be a separate node with more complete syntactic
// information.
func FindHoverContext(ctx context.Context, s Snapshot, pkg Package, obj types.Object, pkgNode ast.Node, fullDecl ast.Decl) (*HoverContext, error) {
	var info *HoverContext

	// Type parameters get their signature from their declaration object.
	if _, isTypeName := obj.(*types.TypeName); isTypeName {
		if _, isTypeParam := obj.Type().(*typeparams.TypeParam); isTypeParam {
			return &HoverContext{signatureSource: obj}, nil
		}
	}

	// This is problematic for a number of reasons. We really need to have a more
	// general mechanism to validate the coherency of AST with type information,
	// but absent that we must do our best to ensure that we don't use fullNode
	// when we actually need the node that was type checked.
	//
	// pkgNode may be nil, if it was eliminated from the type-checked syntax. In
	// that case, use fullDecl if available.
	node := pkgNode
	if node == nil && fullDecl != nil {
		node = fullDecl
	}

	switch node := node.(type) {
	case *ast.Ident:
		// The package declaration.
		for _, f := range pkg.GetSyntax() {
			if f.Name == pkgNode {
				info = &HoverContext{Comment: f.Doc}
			}
		}
	case *ast.ImportSpec:
		// Try to find the package documentation for an imported package.
		importPath, err := strconv.Unquote(node.Path.Value)
		if err != nil {
			return nil, err
		}
		imp, err := pkg.ResolveImportPath(ImportPath(importPath))
		if err != nil {
			return nil, err
		}
		// Assume that only one file will contain package documentation,
		// so pick the first file that has a doc comment.
		for _, file := range imp.GetSyntax() {
			if file.Doc != nil {
				info = &HoverContext{Comment: file.Doc}
				if file.Name != nil {
					info.signatureSource = "package " + file.Name.Name
				}
				break
			}
		}
	case *ast.GenDecl:
		switch obj := obj.(type) {
		case *types.TypeName, *types.Var, *types.Const, *types.Func:
			// Always use the full declaration here if we have it, because the
			// dependent code doesn't rely on pointer identity. This is fragile.
			if d, _ := fullDecl.(*ast.GenDecl); d != nil {
				node = d
			}
			// obj may not have been produced by type checking the AST containing
			// node, so we need to be careful about using token.Pos.
			tok := pkg.FileSet().File(obj.Pos())
			offset, err := safetoken.Offset(tok, obj.Pos())
			if err != nil {
				return nil, err
			}

			// fullTok and fullPos are the *token.File and object position in for the
			// full AST.
			fullTok := pkg.FileSet().File(node.Pos())
			fullPos, err := safetoken.Pos(fullTok, offset)
			if err != nil {
				return nil, err
			}

			var spec ast.Spec
			for _, s := range node.Specs {
				// Avoid panics by guarding the calls to token.Offset (golang/go#48249).
				start, err := safetoken.Offset(fullTok, s.Pos())
				if err != nil {
					return nil, err
				}
				end, err := safetoken.Offset(fullTok, s.End())
				if err != nil {
					return nil, err
				}
				if start <= offset && offset <= end {
					spec = s
					break
				}
			}

			info, err = hoverGenDecl(node, spec, fullPos, obj)
			if err != nil {
				return nil, err
			}
		}
	case *ast.TypeSpec:
		if obj.Parent() == types.Universe {
			if genDecl, ok := fullDecl.(*ast.GenDecl); ok {
				info = hoverTypeSpec(node, genDecl)
			}
		}
	case *ast.FuncDecl:
		switch obj.(type) {
		case *types.Func:
			info = &HoverContext{signatureSource: obj, Comment: node.Doc}
		case *types.Builtin:
			info = &HoverContext{Comment: node.Doc}
			if sig, err := NewBuiltinSignature(ctx, s, obj.Name()); err == nil {
				info.signatureSource = "func " + sig.name + sig.Format()
			} else {
				// Fall back on the object as a signature source.
				bug.Report("invalid builtin hover", bug.Data{
					"err": err.Error(),
				})
				info.signatureSource = obj
			}
		case *types.Var:
			// Object is a function param or the field of an anonymous struct
			// declared with ':='. Skip the first one because only fields
			// can have docs.
			if isFunctionParam(obj, node) {
				break
			}

			_, field := FindDeclAndField(pkg.GetSyntax(), obj.Pos())
			if field != nil {
				comment := field.Doc
				if comment.Text() == "" {
					comment = field.Comment
				}
				info = &HoverContext{signatureSource: obj, Comment: comment}
			}
		}
	}

	if info == nil {
		info = &HoverContext{signatureSource: obj}
	}

	return info, nil
}

// isFunctionParam returns true if the passed object is either an incoming
// or an outgoing function param
func isFunctionParam(obj types.Object, node *ast.FuncDecl) bool {
	for _, f := range node.Type.Params.List {
		if f.Pos() == obj.Pos() {
			return true
		}
	}
	if node.Type.Results != nil {
		for _, f := range node.Type.Results.List {
			if f.Pos() == obj.Pos() {
				return true
			}
		}
	}
	return false
}

// hoverGenDecl returns hover information an object declared via spec inside
// of the GenDecl node. obj is the type-checked object corresponding to the
// declaration, but may have been type-checked using a different AST than the
// given nodes; fullPos is the position of obj in node's AST.
func hoverGenDecl(node *ast.GenDecl, spec ast.Spec, fullPos token.Pos, obj types.Object) (*HoverContext, error) {
	if spec == nil {
		return nil, fmt.Errorf("no spec for node %v at position %v", node, fullPos)
	}

	// If we have a field or method.
	switch obj.(type) {
	case *types.Var, *types.Const, *types.Func:
		return hoverVar(spec, fullPos, obj, node), nil
	}
	// Handle types.
	switch spec := spec.(type) {
	case *ast.TypeSpec:
		return hoverTypeSpec(spec, node), nil
	case *ast.ValueSpec:
		return &HoverContext{signatureSource: spec, Comment: spec.Doc}, nil
	case *ast.ImportSpec:
		return &HoverContext{signatureSource: spec, Comment: spec.Doc}, nil
	}
	return nil, fmt.Errorf("unable to format spec %v (%T)", spec, spec)
}

// TODO(rfindley): rename this function.
func hoverTypeSpec(spec *ast.TypeSpec, decl *ast.GenDecl) *HoverContext {
	comment := spec.Doc
	if comment == nil && decl != nil {
		comment = decl.Doc
	}
	if comment == nil {
		comment = spec.Comment
	}
	return &HoverContext{
		signatureSource: spec,
		Comment:         comment,
	}
}

func hoverVar(node ast.Spec, fullPos token.Pos, obj types.Object, decl *ast.GenDecl) *HoverContext {
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
		// Try to extract the field list of an anonymous struct
		if fieldList = extractFieldList(spec.Type); fieldList != nil {
			break
		}

		comment := spec.Doc
		if comment == nil {
			comment = decl.Doc
		}
		if comment == nil {
			comment = spec.Comment
		}

		// We need the AST nodes for variable declarations of basic literals with
		// associated values so that we can augment their hover with more information.
		if _, ok := obj.(*types.Var); ok && spec.Type == nil && len(spec.Values) > 0 {
			if _, ok := spec.Values[0].(*ast.BasicLit); ok {
				return &HoverContext{signatureSource: spec, Comment: comment}
			}
		}

		return &HoverContext{signatureSource: obj, Comment: comment}
	}

	if fieldList != nil {
		comment := findFieldComment(fullPos, fieldList)
		return &HoverContext{signatureSource: obj, Comment: comment}
	}
	return &HoverContext{signatureSource: obj, Comment: decl.Doc}
}

// extractFieldList recursively tries to extract a field list.
// If it is not found, nil is returned.
func extractFieldList(specType ast.Expr) *ast.FieldList {
	switch t := specType.(type) {
	case *ast.StructType:
		return t.Fields
	case *ast.InterfaceType:
		return t.Methods
	case *ast.ArrayType:
		return extractFieldList(t.Elt)
	case *ast.MapType:
		// Map value has a greater chance to be a struct
		if fields := extractFieldList(t.Value); fields != nil {
			return fields
		}
		return extractFieldList(t.Key)
	case *ast.ChanType:
		return extractFieldList(t.Value)
	}
	return nil
}

// findFieldComment visits all fields in depth-first order and returns
// the comment of a field with passed position. If no comment is found,
// nil is returned.
func findFieldComment(pos token.Pos, fieldList *ast.FieldList) *ast.CommentGroup {
	for _, field := range fieldList.List {
		if field.Pos() == pos {
			if field.Doc.Text() != "" {
				return field.Doc
			}
			return field.Comment
		}

		if nestedFieldList := extractFieldList(field.Type); nestedFieldList != nil {
			if c := findFieldComment(pos, nestedFieldList); c != nil {
				return c
			}
		}
	}
	return nil
}

func FormatHover(h *HoverJSON, options *Options) (string, error) {
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

// BuildLink constructs a link with the given target, path, and anchor.
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
		return CommentToMarkdown(doc)
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

// FindDeclAndField returns the var/func/type/const Decl that declares
// the identifier at pos, searching the given list of file syntax
// trees. If pos is the position of an ast.Field or one of its Names
// or Ellipsis.Elt, the field is returned, along with the innermost
// enclosing Decl, which could be only loosely related---consider:
//
//	var decl = f(  func(field int) {}  )
//
// It returns (nil, nil) if no Field or Decl is found at pos.
func FindDeclAndField(files []*ast.File, pos token.Pos) (decl ast.Decl, field *ast.Field) {
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
			checkField := func(f ast.Node) {
				if f.Pos() == pos {
					field = n
					for i := len(stack) - 1; i >= 0; i-- {
						if d, ok := stack[i].(ast.Decl); ok {
							decl = d // innermost enclosing decl
							break
						}
					}
					panic(found{})
				}
			}

			// Check *ast.Field itself. This handles embedded
			// fields which have no associated *ast.Ident name.
			checkField(n)

			// Check each field name since you can have
			// multiple names for the same type expression.
			for _, name := range n.Names {
				checkField(name)
			}

			// Also check "X" in "...X". This makes it easy
			// to format variadic signature params properly.
			if ell, ok := n.Type.(*ast.Ellipsis); ok && ell.Elt != nil {
				checkField(ell.Elt)
			}

		case *ast.FuncDecl:
			if n.Name.Pos() == pos {
				decl = n
				panic(found{})
			}

		case *ast.GenDecl:
			for _, spec := range n.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					if spec.Name.Pos() == pos {
						decl = n
						panic(found{})
					}
				case *ast.ValueSpec:
					for _, id := range spec.Names {
						if id.Pos() == pos {
							decl = n
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

	return nil, nil
}
