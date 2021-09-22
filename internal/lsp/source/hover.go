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
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"golang.org/x/text/unicode/runenames"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/typeparams"
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

	// LinkPath is the pkg.go.dev link for the given symbol.
	// For example, the "go/ast" part of "pkg.go.dev/go/ast#Node".
	LinkPath string `json:"linkPath"`

	// LinkAnchor is the pkg.go.dev link anchor for the given symbol.
	// For example, the "Node" part of "pkg.go.dev/go/ast#Node".
	LinkAnchor string `json:"linkAnchor"`

	// importPath is the import path for the package containing the given
	// symbol.
	importPath string

	// symbolName is the types.Object.Name for the given symbol.
	symbolName string

	source  interface{}
	comment *ast.CommentGroup

	// typeName contains the identifier name when the identifier is a type declaration.
	// If it is not empty, the hover will have the prefix "type <typeName> ".
	typeName string
	// isTypeAlias indicates whether the identifier is a type alias declaration.
	// If it is true, the hover will have the prefix "type <typeName> = ".
	isTypeAlias bool
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
	// See golang/go#36998: don't link to modules matching GOPRIVATE.
	if snapshot.View().IsGoPrivatePath(h.importPath) {
		h.LinkPath = ""
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
	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return 0, MappedRange{}, err
	}
	spn, err := pgf.Mapper.PointSpan(position)
	if err != nil {
		return 0, MappedRange{}, err
	}
	rng, err := spn.Range(pgf.Mapper.Converter)
	if err != nil {
		return 0, MappedRange{}, err
	}
	pos := rng.Start

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
		litOffset := pgf.Tok.Offset(lit.Pos())
		offset := pgf.Tok.Offset(pos)
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

	mappedRange, err := posToMappedRange(snapshot, pkg, start, end)
	if err != nil {
		return 0, MappedRange{}, err
	}
	return r, mappedRange, nil
}

func HoverIdentifier(ctx context.Context, i *IdentifierInfo) (*HoverInformation, error) {
	ctx, done := event.Start(ctx, "source.Hover")
	defer done()

	fset := i.Snapshot.FileSet()
	h, err := HoverInfo(ctx, i.Snapshot, i.pkg, i.Declaration.obj, i.Declaration.node, i.Declaration.fullDecl)
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
		if h.typeName != "" {
			prefix := "type " + h.typeName + " "
			if h.isTypeAlias {
				prefix += "= "
			}
			h.Signature = prefix + h.Signature
		}

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
	switch obj := obj.(type) {
	case *types.PkgName:
		h.importPath = obj.Imported().Path()
		h.LinkPath = h.importPath
		h.symbolName = obj.Name()
		if mod, version, ok := moduleAtVersion(h.LinkPath, i); ok {
			h.LinkPath = strings.Replace(h.LinkPath, mod, mod+"@"+version, 1)
		}
		return h, nil
	case *types.Builtin:
		h.importPath = "builtin"
		h.LinkPath = h.importPath
		h.LinkAnchor = obj.Name()
		h.symbolName = h.LinkAnchor
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
	// Don't return links for other unexported types.
	if !obj.Exported() {
		return h, nil
	}
	var rTypeName string
	switch obj := obj.(type) {
	case *types.Var:
		// If the object is a field, and we have an associated selector
		// composite literal, or struct, we can determine the link.
		if obj.IsField() {
			if named, ok := i.enclosing.(*types.Named); ok {
				rTypeName = named.Obj().Name()
			}
		}
	case *types.Func:
		typ, ok := obj.Type().(*types.Signature)
		if !ok {
			return h, nil
		}
		if r := typ.Recv(); r != nil {
			switch rtyp := Deref(r.Type()).(type) {
			case *types.Struct:
				rTypeName = r.Name()
			case *types.Named:
				// If we have an unexported type, see if the enclosing type is
				// exported (we may have an interface or struct we can link
				// to). If not, don't show any link.
				if !rtyp.Obj().Exported() {
					if named, ok := i.enclosing.(*types.Named); ok && named.Obj().Exported() {
						rTypeName = named.Obj().Name()
					} else {
						return h, nil
					}
				} else {
					rTypeName = rtyp.Obj().Name()
				}
			}
		}
	}
	if obj.Pkg() == nil {
		event.Log(ctx, fmt.Sprintf("nil package for %s", obj))
		return h, nil
	}
	h.importPath = obj.Pkg().Path()
	h.LinkPath = h.importPath
	if mod, version, ok := moduleAtVersion(h.LinkPath, i); ok {
		h.LinkPath = strings.Replace(h.LinkPath, mod, mod+"@"+version, 1)
	}
	if rTypeName != "" {
		h.LinkAnchor = fmt.Sprintf("%s.%s", rTypeName, obj.Name())
		h.symbolName = fmt.Sprintf("(%s.%s).%s", obj.Pkg().Name(), rTypeName, obj.Name())
		return h, nil
	}
	// For most cases, the link is "package/path#symbol".
	h.LinkAnchor = obj.Name()
	h.symbolName = fmt.Sprintf("%s.%s", obj.Pkg().Name(), obj.Name())
	return h, nil
}

func moduleAtVersion(path string, i *IdentifierInfo) (string, string, bool) {
	if strings.ToLower(i.Snapshot.View().Options().LinkTarget) != "pkg.go.dev" {
		return "", "", false
	}
	impPkg, err := i.pkg.GetImport(path)
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

// HoverInfo returns a HoverInformation struct for an ast node and its type
// object. node should be the actual node used in type checking, while fullNode
// could be a separate node with more complete syntactic information.
func HoverInfo(ctx context.Context, s Snapshot, pkg Package, obj types.Object, pkgNode ast.Node, fullDecl ast.Decl) (*HoverInformation, error) {
	var info *HoverInformation

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
				info = &HoverInformation{comment: f.Doc}
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
			for _, file := range imp.GetSyntax() {
				if file.Doc != nil {
					info = &HoverInformation{source: obj, comment: file.Doc}
					break
				}
			}
		}
		info = &HoverInformation{source: node}
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
			tok := s.FileSet().File(obj.Pos())
			offset := tok.Offset(obj.Pos())
			tok2 := s.FileSet().File(node.Pos())
			var spec ast.Spec
			for _, s := range node.Specs {
				// Avoid panics by guarding the calls to token.Offset (golang/go#48249).
				if InRange(tok2, s.Pos()) && InRange(tok2, s.End()) && tok2.Offset(s.Pos()) <= offset && offset <= tok2.Offset(s.End()) {
					spec = s
					break
				}
			}
			var err error
			info, err = formatGenDecl(node, spec, obj, obj.Type())
			if err != nil {
				return nil, err
			}
		}
	case *ast.TypeSpec:
		if obj.Parent() == types.Universe {
			if obj.Name() == "error" {
				info = &HoverInformation{source: node}
			} else {
				info = &HoverInformation{source: node.Name} // comments not needed for builtins
			}
		}
	case *ast.FuncDecl:
		switch obj.(type) {
		case *types.Func:
			info = &HoverInformation{source: obj, comment: node.Doc}
		case *types.Builtin:
			info = &HoverInformation{source: node.Type, comment: node.Doc}
		case *types.Var:
			// Object is a function param or the field of an anonymous struct
			// declared with ':='. Skip the first one because only fields
			// can have docs.
			if isFunctionParam(obj, node) {
				break
			}

			field, err := s.PosToField(ctx, pkg, obj.Pos())
			if err != nil {
				return nil, err
			}

			if field != nil {
				comment := field.Doc
				if comment.Text() == "" {
					comment = field.Comment
				}
				info = &HoverInformation{source: obj, comment: comment}
			}
		}
	}

	if info == nil {
		info = &HoverInformation{source: obj}
	}

	if info.comment != nil {
		info.FullDocumentation = info.comment.Text()
		info.Synopsis = doc.Synopsis(info.FullDocumentation)
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

func formatGenDecl(node *ast.GenDecl, spec ast.Spec, obj types.Object, typ types.Type) (*HoverInformation, error) {
	if _, ok := typ.(*types.Named); ok {
		switch typ.Underlying().(type) {
		case *types.Interface, *types.Struct:
			return formatGenDecl(node, spec, obj, typ.Underlying())
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
		return formatTypeSpec(spec, node), nil
	case *ast.ValueSpec:
		return &HoverInformation{source: spec, comment: spec.Doc}, nil
	case *ast.ImportSpec:
		return &HoverInformation{source: spec, comment: spec.Doc}, nil
	}
	return nil, errors.Errorf("unable to format spec %v (%T)", spec, spec)
}

func formatTypeSpec(spec *ast.TypeSpec, decl *ast.GenDecl) *HoverInformation {
	comment := spec.Doc
	if comment == nil && decl != nil {
		comment = decl.Doc
	}
	if comment == nil {
		comment = spec.Comment
	}
	return &HoverInformation{
		source:      spec.Type,
		comment:     comment,
		typeName:    spec.Name.Name,
		isTypeAlias: spec.Assign.IsValid(),
	}
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
				return &HoverInformation{source: spec, comment: comment}
			}
		}

		return &HoverInformation{source: obj, comment: comment}
	}

	if fieldList != nil {
		comment := findFieldComment(obj.Pos(), fieldList)
		return &HoverInformation{source: obj, comment: comment}
	}
	return &HoverInformation{source: obj, comment: decl.Doc}
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

func FormatHover(h *HoverInformation, options *Options) (string, error) {
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
	link := formatLink(h, options)
	switch options.HoverKind {
	case SynopsisDocumentation:
		doc := formatDoc(h.Synopsis, options)
		return formatHover(options, signature, link, doc), nil
	case FullDocumentation:
		doc := formatDoc(h.FullDocumentation, options)
		return formatHover(options, signature, link, doc), nil
	}
	return "", errors.Errorf("no hover for %v", h.source)
}

func formatLink(h *HoverInformation, options *Options) string {
	if !options.LinksInHover || options.LinkTarget == "" || h.LinkPath == "" {
		return ""
	}
	plainLink := BuildLink(options.LinkTarget, h.LinkPath, h.LinkAnchor)
	switch options.PreferredContentFormat {
	case protocol.Markdown:
		return fmt.Sprintf("[`%s` on %s](%s)", h.symbolName, options.LinkTarget, plainLink)
	case protocol.PlainText:
		return ""
	default:
		return plainLink
	}
}

// BuildLink constructs a link with the given target, path, and anchor.
func BuildLink(target, path, anchor string) string {
	link := fmt.Sprintf("https://%s/%s", target, path)
	if target == "pkg.go.dev" {
		link += "?utm_source=gopls"
	}
	if anchor == "" {
		return link
	}
	return link + "#" + anchor
}

func formatDoc(doc string, options *Options) string {
	if options.PreferredContentFormat == protocol.Markdown {
		return CommentToMarkdown(doc)
	}
	return doc
}

func formatHover(options *Options, x ...string) string {
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
