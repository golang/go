// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fillstruct defines an Analyzer that automatically
// fills in a struct declaration with zero value elements for each field.
//
// The analyzer's diagnostic is merely a prompt.
// The actual fix is created by a separate direct call from gopls to
// the SuggestedFixes function.
// Tests of Analyzer.Run can be found in ./testdata/src.
// Tests of the SuggestedFixes logic live in ../../testdata/fillstruct.
package fillstruct

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/fuzzy"
	"golang.org/x/tools/internal/typeparams"
)

const Doc = `note incomplete struct initializations

This analyzer provides diagnostics for any struct literals that do not have
any fields initialized. Because the suggested fix for this analysis is
expensive to compute, callers should compute it separately, using the
SuggestedFix function below.
`

var Analyzer = &analysis.Analyzer{
	Name:             "fillstruct",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	Run:              run,
	RunDespiteErrors: true,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{(*ast.CompositeLit)(nil)}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		expr := n.(*ast.CompositeLit)

		// Find enclosing file.
		// TODO(adonovan): use inspect.WithStack?
		var file *ast.File
		for _, f := range pass.Files {
			if f.Pos() <= expr.Pos() && expr.Pos() <= f.End() {
				file = f
				break
			}
		}
		if file == nil {
			return
		}

		typ := pass.TypesInfo.TypeOf(expr)
		if typ == nil {
			return
		}

		// Find reference to the type declaration of the struct being initialized.
		typ = deref(typ)
		tStruct, ok := typ.Underlying().(*types.Struct)
		if !ok {
			return
		}
		// Inv: typ is the possibly-named struct type.

		fieldCount := tStruct.NumFields()

		// Skip any struct that is already populated or that has no fields.
		if fieldCount == 0 || fieldCount == len(expr.Elts) {
			return
		}

		// Are any fields in need of filling?
		var fillableFields []string
		for i := 0; i < fieldCount; i++ {
			field := tStruct.Field(i)
			// Ignore fields that are not accessible in the current package.
			if field.Pkg() != nil && field.Pkg() != pass.Pkg && !field.Exported() {
				continue
			}
			fillableFields = append(fillableFields, fmt.Sprintf("%s: %s", field.Name(), field.Type().String()))
		}
		if len(fillableFields) == 0 {
			return
		}

		// Derive a name for the struct type.
		var name string
		if typ != tStruct {
			// named struct type (e.g. pkg.S[T])
			name = types.TypeString(typ, types.RelativeTo(pass.Pkg))
		} else {
			// anonymous struct type
			totalFields := len(fillableFields)
			const maxLen = 20
			// Find the index to cut off printing of fields.
			var i, fieldLen int
			for i = range fillableFields {
				if fieldLen > maxLen {
					break
				}
				fieldLen += len(fillableFields[i])
			}
			fillableFields = fillableFields[:i]
			if i < totalFields {
				fillableFields = append(fillableFields, "...")
			}
			name = fmt.Sprintf("anonymous struct { %s }", strings.Join(fillableFields, ", "))
		}
		pass.Report(analysis.Diagnostic{
			Message: fmt.Sprintf("Fill %s", name),
			Pos:     expr.Pos(),
			End:     expr.End(),
		})
	})
	return nil, nil
}

// SuggestedFix computes the suggested fix for the kinds of
// diagnostics produced by the Analyzer above.
func SuggestedFix(fset *token.FileSet, start, end token.Pos, content []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error) {
	if info == nil {
		return nil, fmt.Errorf("nil types.Info")
	}

	pos := start // don't use the end

	// TODO(rstambler): Using ast.Inspect would probably be more efficient than
	// calling PathEnclosingInterval. Switch this approach.
	path, _ := astutil.PathEnclosingInterval(file, pos, pos)
	if len(path) == 0 {
		return nil, fmt.Errorf("no enclosing ast.Node")
	}
	var expr *ast.CompositeLit
	for _, n := range path {
		if node, ok := n.(*ast.CompositeLit); ok {
			expr = node
			break
		}
	}

	typ := info.TypeOf(expr)
	if typ == nil {
		return nil, fmt.Errorf("no composite literal")
	}

	// Find reference to the type declaration of the struct being initialized.
	typ = deref(typ)
	tStruct, ok := typ.Underlying().(*types.Struct)
	if !ok {
		return nil, fmt.Errorf("%s is not a (pointer to) struct type",
			types.TypeString(typ, types.RelativeTo(pkg)))
	}
	// Inv: typ is the possibly-named struct type.

	fieldCount := tStruct.NumFields()

	// Check which types have already been filled in. (we only want to fill in
	// the unfilled types, or else we'll blat user-supplied details)
	prefilledFields := map[string]ast.Expr{}
	for _, e := range expr.Elts {
		if kv, ok := e.(*ast.KeyValueExpr); ok {
			if key, ok := kv.Key.(*ast.Ident); ok {
				prefilledFields[key.Name] = kv.Value
			}
		}
	}

	// Use a new fileset to build up a token.File for the new composite
	// literal. We need one line for foo{, one line for }, and one line for
	// each field we're going to set. format.Node only cares about line
	// numbers, so we don't need to set columns, and each line can be
	// 1 byte long.
	// TODO(adonovan): why is this necessary? The position information
	// is going to be wrong for the existing trees in prefilledFields.
	// Can't the formatter just do its best with an empty fileset?
	fakeFset := token.NewFileSet()
	tok := fakeFset.AddFile("", -1, fieldCount+2)

	line := 2 // account for 1-based lines and the left brace
	var fieldTyps []types.Type
	for i := 0; i < fieldCount; i++ {
		field := tStruct.Field(i)
		// Ignore fields that are not accessible in the current package.
		if field.Pkg() != nil && field.Pkg() != pkg && !field.Exported() {
			fieldTyps = append(fieldTyps, nil)
			continue
		}
		fieldTyps = append(fieldTyps, field.Type())
	}
	matches := analysisinternal.MatchingIdents(fieldTyps, file, start, info, pkg)
	var elts []ast.Expr
	for i, fieldTyp := range fieldTyps {
		if fieldTyp == nil {
			continue // TODO(adonovan): is this reachable?
		}
		fieldName := tStruct.Field(i).Name()

		tok.AddLine(line - 1) // add 1 byte per line
		if line > tok.LineCount() {
			panic(fmt.Sprintf("invalid line number %v (of %v) for fillstruct", line, tok.LineCount()))
		}
		pos := tok.LineStart(line)

		kv := &ast.KeyValueExpr{
			Key: &ast.Ident{
				NamePos: pos,
				Name:    fieldName,
			},
			Colon: pos,
		}
		if expr, ok := prefilledFields[fieldName]; ok {
			kv.Value = expr
		} else {
			names, ok := matches[fieldTyp]
			if !ok {
				return nil, fmt.Errorf("invalid struct field type: %v", fieldTyp)
			}

			// Find the name most similar to the field name.
			// If no name matches the pattern, generate a zero value.
			// NOTE: We currently match on the name of the field key rather than the field type.
			if best := fuzzy.BestMatch(fieldName, names); best != "" {
				kv.Value = ast.NewIdent(best)
			} else if v := populateValue(file, pkg, fieldTyp); v != nil {
				kv.Value = v
			} else {
				return nil, nil
			}
		}
		elts = append(elts, kv)
		line++
	}

	// If all of the struct's fields are unexported, we have nothing to do.
	if len(elts) == 0 {
		return nil, fmt.Errorf("no elements to fill")
	}

	// Add the final line for the right brace. Offset is the number of
	// bytes already added plus 1.
	tok.AddLine(len(elts) + 1)
	line = len(elts) + 2
	if line > tok.LineCount() {
		panic(fmt.Sprintf("invalid line number %v (of %v) for fillstruct", line, tok.LineCount()))
	}

	cl := &ast.CompositeLit{
		Type:   expr.Type,
		Lbrace: tok.LineStart(1),
		Elts:   elts,
		Rbrace: tok.LineStart(line),
	}

	// Find the line on which the composite literal is declared.
	split := bytes.Split(content, []byte("\n"))
	lineNumber := safetoken.StartPosition(fset, expr.Lbrace).Line
	firstLine := split[lineNumber-1] // lines are 1-indexed

	// Trim the whitespace from the left of the line, and use the index
	// to get the amount of whitespace on the left.
	trimmed := bytes.TrimLeftFunc(firstLine, unicode.IsSpace)
	index := bytes.Index(firstLine, trimmed)
	whitespace := firstLine[:index]

	// First pass through the formatter: turn the expr into a string.
	var formatBuf bytes.Buffer
	if err := format.Node(&formatBuf, fakeFset, cl); err != nil {
		return nil, fmt.Errorf("failed to run first format on:\n%s\ngot err: %v", cl.Type, err)
	}
	sug := indent(formatBuf.Bytes(), whitespace)

	if len(prefilledFields) > 0 {
		// Attempt a second pass through the formatter to line up columns.
		sourced, err := format.Source(sug)
		if err == nil {
			sug = indent(sourced, whitespace)
		}
	}

	return &analysis.SuggestedFix{
		TextEdits: []analysis.TextEdit{
			{
				Pos:     expr.Pos(),
				End:     expr.End(),
				NewText: sug,
			},
		},
	}, nil
}

// indent works line by line through str, indenting (prefixing) each line with
// ind.
func indent(str, ind []byte) []byte {
	split := bytes.Split(str, []byte("\n"))
	newText := bytes.NewBuffer(nil)
	for i, s := range split {
		if len(s) == 0 {
			continue
		}
		// Don't add the extra indentation to the first line.
		if i != 0 {
			newText.Write(ind)
		}
		newText.Write(s)
		if i < len(split)-1 {
			newText.WriteByte('\n')
		}
	}
	return newText.Bytes()
}

// populateValue constructs an expression to fill the value of a struct field.
//
// When the type of a struct field is a basic literal or interface, we return
// default values. For other types, such as maps, slices, and channels, we create
// empty expressions such as []T{} or make(chan T) rather than using default values.
//
// The reasoning here is that users will call fillstruct with the intention of
// initializing the struct, in which case setting these fields to nil has no effect.
func populateValue(f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	switch u := typ.Underlying().(type) {
	case *types.Basic:
		switch {
		case u.Info()&types.IsNumeric != 0:
			return &ast.BasicLit{Kind: token.INT, Value: "0"}
		case u.Info()&types.IsBoolean != 0:
			return &ast.Ident{Name: "false"}
		case u.Info()&types.IsString != 0:
			return &ast.BasicLit{Kind: token.STRING, Value: `""`}
		case u.Kind() == types.UnsafePointer:
			return ast.NewIdent("nil")
		default:
			panic("unknown basic type")
		}

	case *types.Map:
		k := analysisinternal.TypeExpr(f, pkg, u.Key())
		v := analysisinternal.TypeExpr(f, pkg, u.Elem())
		if k == nil || v == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: &ast.MapType{
				Key:   k,
				Value: v,
			},
		}
	case *types.Slice:
		s := analysisinternal.TypeExpr(f, pkg, u.Elem())
		if s == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: &ast.ArrayType{
				Elt: s,
			},
		}

	case *types.Array:
		a := analysisinternal.TypeExpr(f, pkg, u.Elem())
		if a == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: &ast.ArrayType{
				Elt: a,
				Len: &ast.BasicLit{
					Kind: token.INT, Value: fmt.Sprintf("%v", u.Len()),
				},
			},
		}

	case *types.Chan:
		v := analysisinternal.TypeExpr(f, pkg, u.Elem())
		if v == nil {
			return nil
		}
		dir := ast.ChanDir(u.Dir())
		if u.Dir() == types.SendRecv {
			dir = ast.SEND | ast.RECV
		}
		return &ast.CallExpr{
			Fun: ast.NewIdent("make"),
			Args: []ast.Expr{
				&ast.ChanType{
					Dir:   dir,
					Value: v,
				},
			},
		}

	case *types.Struct:
		s := analysisinternal.TypeExpr(f, pkg, typ)
		if s == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: s,
		}

	case *types.Signature:
		var params []*ast.Field
		for i := 0; i < u.Params().Len(); i++ {
			p := analysisinternal.TypeExpr(f, pkg, u.Params().At(i).Type())
			if p == nil {
				return nil
			}
			params = append(params, &ast.Field{
				Type: p,
				Names: []*ast.Ident{
					{
						Name: u.Params().At(i).Name(),
					},
				},
			})
		}
		var returns []*ast.Field
		for i := 0; i < u.Results().Len(); i++ {
			r := analysisinternal.TypeExpr(f, pkg, u.Results().At(i).Type())
			if r == nil {
				return nil
			}
			returns = append(returns, &ast.Field{
				Type: r,
			})
		}
		return &ast.FuncLit{
			Type: &ast.FuncType{
				Params: &ast.FieldList{
					List: params,
				},
				Results: &ast.FieldList{
					List: returns,
				},
			},
			Body: &ast.BlockStmt{},
		}

	case *types.Pointer:
		switch u.Elem().(type) {
		case *types.Basic:
			return &ast.CallExpr{
				Fun: &ast.Ident{
					Name: "new",
				},
				Args: []ast.Expr{
					&ast.Ident{
						Name: u.Elem().String(),
					},
				},
			}
		default:
			return &ast.UnaryExpr{
				Op: token.AND,
				X:  populateValue(f, pkg, u.Elem()),
			}
		}

	case *types.Interface:
		if param, ok := typ.(*typeparams.TypeParam); ok {
			// *new(T) is the zero value of a type parameter T.
			// TODO(adonovan): one could give a more specific zero
			// value if the type has a core type that is, say,
			// always a number or a pointer. See go/ssa for details.
			return &ast.StarExpr{
				X: &ast.CallExpr{
					Fun: ast.NewIdent("new"),
					Args: []ast.Expr{
						ast.NewIdent(param.Obj().Name()),
					},
				},
			}
		}

		return ast.NewIdent("nil")
	}
	return nil
}

func deref(t types.Type) types.Type {
	for {
		ptr, ok := t.Underlying().(*types.Pointer)
		if !ok {
			return t
		}
		t = ptr.Elem()
	}
}
