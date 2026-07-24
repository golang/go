// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command toolsvendor reapplies enum AST compatibility changes to the x/tools
// packages in cmd/vendor after "go mod vendor" replaces them.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type edit struct {
	path string
	old  string
	new  string
}

var edits = []edit{
	{
		"golang.org/x/tools/go/ast/edge/edge.go",
		`\tValueSpec_Names
\tValueSpec_Type
\tValueSpec_Values

\tmaxKind`,
		`\tValueSpec_Names
\tValueSpec_Type
\tValueSpec_Values
\tEnumDecl_Doc
\tEnumDecl_Name
\tEnumDecl_TypeParams
\tEnumDecl_Variants
\tEnumVariant_Comment
\tEnumVariant_Doc
\tEnumVariant_Fields
\tEnumVariant_Name

\tmaxKind`,
	},
	{
		"golang.org/x/tools/go/ast/edge/edge.go",
		`\tValueSpec_Names:       info[*ast.ValueSpec]("Names"),
\tValueSpec_Type:        info[*ast.ValueSpec]("Type"),
\tValueSpec_Values:      info[*ast.ValueSpec]("Values"),
}`,
		`\tValueSpec_Names:       info[*ast.ValueSpec]("Names"),
\tValueSpec_Type:        info[*ast.ValueSpec]("Type"),
\tValueSpec_Values:      info[*ast.ValueSpec]("Values"),
\tEnumDecl_Doc:          info[*ast.EnumDecl]("Doc"),
\tEnumDecl_Name:         info[*ast.EnumDecl]("Name"),
\tEnumDecl_TypeParams:   info[*ast.EnumDecl]("TypeParams"),
\tEnumDecl_Variants:     info[*ast.EnumDecl]("Variants"),
\tEnumVariant_Comment:   info[*ast.EnumVariant]("Comment"),
\tEnumVariant_Doc:       info[*ast.EnumVariant]("Doc"),
\tEnumVariant_Fields:    info[*ast.EnumVariant]("Fields"),
\tEnumVariant_Name:      info[*ast.EnumVariant]("Name"),
}`,
	},
	{
		"golang.org/x/tools/go/ast/inspector/typeof.go",
		`\tnTypeSwitchStmt
\tnUnaryExpr
\tnValueSpec
)`,
		`\tnTypeSwitchStmt
\tnUnaryExpr
\tnValueSpec
\tnEnumDecl
\tnEnumVariant
)`,
	},
	{
		"golang.org/x/tools/go/ast/inspector/typeof.go",
		`\tcase *ast.ValueSpec:
\t\treturn 1 << nValueSpec
\t}
\treturn 0`,
		`\tcase *ast.ValueSpec:
\t\treturn 1 << nValueSpec
\tcase *ast.EnumDecl:
\t\treturn 1 << nEnumDecl
\tcase *ast.EnumVariant:
\t\treturn 1 << nEnumVariant
\t}
\treturn 0`,
	},
	{
		"golang.org/x/tools/go/ast/inspector/walk.go",
		`\t\twalkList(v, edge.GenDecl_Specs, n.Specs)

\tcase *ast.FuncDecl:`,
		`\t\twalkList(v, edge.GenDecl_Specs, n.Specs)

\tcase *ast.EnumDecl:
\t\tif n.Doc != nil {
\t\t\twalk(v, edge.EnumDecl_Doc, -1, n.Doc)
\t\t}
\t\twalk(v, edge.EnumDecl_Name, -1, n.Name)
\t\tif n.TypeParams != nil {
\t\t\twalk(v, edge.EnumDecl_TypeParams, -1, n.TypeParams)
\t\t}
\t\twalkList(v, edge.EnumDecl_Variants, n.Variants)

\tcase *ast.EnumVariant:
\t\tif n.Doc != nil {
\t\t\twalk(v, edge.EnumVariant_Doc, -1, n.Doc)
\t\t}
\t\twalk(v, edge.EnumVariant_Name, -1, n.Name)
\t\tif n.Fields != nil {
\t\t\twalk(v, edge.EnumVariant_Fields, -1, n.Fields)
\t\t}
\t\tif n.Comment != nil {
\t\t\twalk(v, edge.EnumVariant_Comment, -1, n.Comment)
\t\t}

\tcase *ast.FuncDecl:`,
	},
	{
		"golang.org/x/tools/go/cfg/builder.go",
		`\tcase *ast.DeclStmt:
\t\t// Treat each var ValueSpec as a separate statement.
\t\td := s.Decl.(*ast.GenDecl)
\t\tif d.Tok == token.VAR {`,
		`\tcase *ast.DeclStmt:
\t\t// Treat each var ValueSpec as a separate statement.
\t\td, ok := s.Decl.(*ast.GenDecl)
\t\tif !ok {
\t\t\tbreak // local enum or another declaration with no control-flow effect
\t\t}
\t\tif d.Tok == token.VAR {`,
	},
	{
		"golang.org/x/tools/internal/refactor/inline/calleefx.go",
		`\t\tcase *ast.DeclStmt:
\t\t\tdecl := n.Decl.(*ast.GenDecl)
\t\t\tfor _, spec := range decl.Specs {`,
		`\t\tcase *ast.DeclStmt:
\t\t\tdecl, ok := n.Decl.(*ast.GenDecl)
\t\t\tif !ok {
\t\t\t\treturn true // local enum declaration has no runtime effect
\t\t\t}
\t\t\tfor _, spec := range decl.Specs {`,
	},
	{
		"golang.org/x/tools/internal/refactor/inline/inline.go",
		`\t\tcase *ast.DeclStmt:
\t\t\tfor _, spec := range stmt.Decl.(*ast.GenDecl).Specs {
\t\t\t\tswitch spec := spec.(type) {
\t\t\t\tcase *ast.ValueSpec:
\t\t\t\t\tfor _, id := range spec.Names {
\t\t\t\t\t\tnames[id.Name] = true
\t\t\t\t\t}
\t\t\t\tcase *ast.TypeSpec:
\t\t\t\t\tnames[spec.Name.Name] = true
\t\t\t\t}
\t\t\t}`,
		`\t\tcase *ast.DeclStmt:
\t\t\tswitch decl := stmt.Decl.(type) {
\t\t\tcase *ast.GenDecl:
\t\t\t\tfor _, spec := range decl.Specs {
\t\t\t\t\tswitch spec := spec.(type) {
\t\t\t\t\tcase *ast.ValueSpec:
\t\t\t\t\t\tfor _, id := range spec.Names {
\t\t\t\t\t\t\tnames[id.Name] = true
\t\t\t\t\t\t}
\t\t\t\t\tcase *ast.TypeSpec:
\t\t\t\t\t\tnames[spec.Name.Name] = true
\t\t\t\t\t}
\t\t\t\t}
\t\t\tcase *ast.EnumDecl:
\t\t\t\tnames[decl.Name.Name] = true
\t\t\t\tfor _, variant := range decl.Variants {
\t\t\t\t\tnames[variant.Name.Name] = true
\t\t\t\t}
\t\t\t}`,
	},
	{
		"golang.org/x/tools/refactor/satisfy/find.go",
		`\tcase *ast.DeclStmt:
\t\td := s.Decl.(*ast.GenDecl)
\t\tif d.Tok == token.VAR { // ignore consts`,
		`\tcase *ast.DeclStmt:
\t\td, ok := s.Decl.(*ast.GenDecl)
\t\tif !ok {
\t\t\tbreak // local enum declaration has no assignment constraints
\t\t}
\t\tif d.Tok == token.VAR { // ignore consts`,
	},
}

func main() {
	vendor := flag.String("vendor", "vendor", "path to the vendor directory")
	flag.Parse()
	for _, edit := range edits {
		if err := apply(filepath.Join(*vendor, filepath.FromSlash(edit.path)), edit); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}

func apply(path string, edit edit) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	s := string(data)
	old := strings.ReplaceAll(edit.old, `\t`, "\t")
	new := strings.ReplaceAll(edit.new, `\t`, "\t")
	if strings.Contains(s, new) {
		return nil
	}
	if strings.Count(s, old) != 1 {
		return fmt.Errorf("%s: x/tools vendor edit no longer applies", path)
	}
	s = strings.Replace(s, old, new, 1)
	return os.WriteFile(path, []byte(s), 0o666)
}
