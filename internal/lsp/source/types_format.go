// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/doc"
	"go/printer"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
)

// formatType returns the detail and kind for a types.Type.
func formatType(typ types.Type, qf types.Qualifier) (detail string, kind protocol.CompletionItemKind) {
	if types.IsInterface(typ) {
		detail = "interface{...}"
		kind = protocol.InterfaceCompletion
	} else if _, ok := typ.(*types.Struct); ok {
		detail = "struct{...}"
		kind = protocol.StructCompletion
	} else if typ != typ.Underlying() {
		detail, kind = formatType(typ.Underlying(), qf)
	} else {
		detail = types.TypeString(typ, qf)
		kind = protocol.ClassCompletion
	}
	return detail, kind
}

type signature struct {
	name, doc        string
	params, results  []string
	variadic         bool
	needResultParens bool
}

func (s *signature) format() string {
	var b strings.Builder
	b.WriteByte('(')
	for i, p := range s.params {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(p)
	}
	b.WriteByte(')')

	// Add space between parameters and results.
	if len(s.results) > 0 {
		b.WriteByte(' ')
	}
	if s.needResultParens {
		b.WriteByte('(')
	}
	for i, r := range s.results {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(r)
	}
	if s.needResultParens {
		b.WriteByte(')')
	}
	return b.String()
}

func newBuiltinSignature(ctx context.Context, view View, name string) (*signature, error) {
	astObj, err := view.LookupBuiltin(ctx, name)
	if err != nil {
		return nil, err
	}
	decl, ok := astObj.Decl.(*ast.FuncDecl)
	if !ok {
		return nil, fmt.Errorf("no function declaration for builtin: %s", name)
	}
	if decl.Type == nil {
		return nil, fmt.Errorf("no type for builtin decl %s", decl.Name)
	}
	var variadic bool
	if decl.Type.Params.List != nil {
		numParams := len(decl.Type.Params.List)
		lastParam := decl.Type.Params.List[numParams-1]
		if _, ok := lastParam.Type.(*ast.Ellipsis); ok {
			variadic = true
		}
	}
	params, _ := formatFieldList(ctx, view, decl.Type.Params, variadic)
	results, needResultParens := formatFieldList(ctx, view, decl.Type.Results, false)
	return &signature{
		doc:              decl.Doc.Text(),
		name:             name,
		needResultParens: needResultParens,
		params:           params,
		results:          results,
		variadic:         variadic,
	}, nil
}

func newSignature(ctx context.Context, s Snapshot, pkg Package, name string, sig *types.Signature, comment *ast.CommentGroup, qf types.Qualifier) (*signature, error) {
	params := make([]string, 0, sig.Params().Len())
	for i := 0; i < sig.Params().Len(); i++ {
		el := sig.Params().At(i)
		typ, err := formatFieldType(ctx, s, pkg, el)
		if err != nil {
			typ = types.TypeString(el.Type(), qf)
		}
		// Handle a variadic parameter (can only be the final parameter).
		if sig.Variadic() && i == sig.Params().Len()-1 {
			typ = strings.Replace(typ, "[]", "...", 1)
		}
		p := typ
		if el.Name() != "" {
			p = el.Name() + " " + typ
		}
		params = append(params, p)
	}
	var needResultParens bool
	results := make([]string, 0, sig.Results().Len())
	for i := 0; i < sig.Results().Len(); i++ {
		if i >= 1 {
			needResultParens = true
		}
		el := sig.Results().At(i)
		typ := types.TypeString(el.Type(), qf)

		if el.Name() == "" {
			results = append(results, typ)
		} else {
			if i == 0 {
				needResultParens = true
			}
			results = append(results, el.Name()+" "+typ)
		}
	}
	var c string
	if comment != nil {
		c = doc.Synopsis(comment.Text())
	}
	return &signature{
		doc:              c,
		params:           params,
		results:          results,
		variadic:         sig.Variadic(),
		needResultParens: needResultParens,
	}, nil
}

func formatFieldType(ctx context.Context, s Snapshot, srcpkg Package, obj *types.Var) (string, error) {
	file, pkg, err := findPosInPackage(s.View(), srcpkg, obj.Pos())
	if err != nil {
		return "", err
	}
	ident, err := findIdentifier(ctx, s, pkg, file, obj.Pos())
	if err != nil {
		return "", err
	}
	if i := ident.ident; i == nil || i.Obj == nil || i.Obj.Decl == nil {
		return "", fmt.Errorf("no object for ident %v", i.Name)
	}
	f, ok := ident.ident.Obj.Decl.(*ast.Field)
	if !ok {
		return "", fmt.Errorf("ident %s is not a field type", ident.Name)
	}
	return formatNode(s.View().Session().Cache().FileSet(), f.Type), nil
}

var replacer = strings.NewReplacer(
	`ComplexType`, `complex128`,
	`FloatType`, `float64`,
	`IntegerType`, `int`,
)

func formatFieldList(ctx context.Context, view View, list *ast.FieldList, variadic bool) ([]string, bool) {
	if list == nil {
		return nil, false
	}
	var writeResultParens bool
	var result []string
	for i := 0; i < len(list.List); i++ {
		if i >= 1 {
			writeResultParens = true
		}
		p := list.List[i]
		cfg := printer.Config{Mode: printer.UseSpaces | printer.TabIndent, Tabwidth: 4}
		b := &bytes.Buffer{}
		if err := cfg.Fprint(b, view.Session().Cache().FileSet(), p.Type); err != nil {
			event.Error(ctx, "unable to print type", nil, tag.Type.Of(p.Type))
			continue
		}
		typ := replacer.Replace(b.String())
		if len(p.Names) == 0 {
			result = append(result, typ)
		}
		for _, name := range p.Names {
			if name.Name != "" {
				if i == 0 {
					writeResultParens = true
				}
				result = append(result, fmt.Sprintf("%s %s", name.Name, typ))
			} else {
				result = append(result, typ)
			}
		}
	}
	if variadic {
		result[len(result)-1] = strings.Replace(result[len(result)-1], "[]", "...", 1)
	}
	return result, writeResultParens
}
