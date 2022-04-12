// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package commandmeta provides metadata about LSP commands, by analyzing the
// command.Interface type.
package commandmeta

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
	"strings"
	"unicode"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/command"
)

type Command struct {
	MethodName string
	Name       string
	// TODO(rFindley): I think Title can actually be eliminated. In all cases
	// where we use it, there is probably a more appropriate contextual title.
	Title  string
	Doc    string
	Args   []*Field
	Result *Field
}

func (c *Command) ID() string {
	return command.ID(c.Name)
}

type Field struct {
	Name     string
	Doc      string
	JSONTag  string
	Type     types.Type
	FieldMod string
	// In some circumstances, we may want to recursively load additional field
	// descriptors for fields of struct types, documenting their internals.
	Fields []*Field
}

func Load() (*packages.Package, []*Command, error) {
	pkgs, err := packages.Load(
		&packages.Config{
			Mode:       packages.NeedTypes | packages.NeedTypesInfo | packages.NeedSyntax | packages.NeedImports | packages.NeedDeps,
			BuildFlags: []string{"-tags=generate"},
		},
		"golang.org/x/tools/internal/lsp/command",
	)
	if err != nil {
		return nil, nil, fmt.Errorf("packages.Load: %v", err)
	}
	pkg := pkgs[0]
	if len(pkg.Errors) > 0 {
		return pkg, nil, pkg.Errors[0]
	}

	// For a bit of type safety, use reflection to get the interface name within
	// the package scope.
	it := reflect.TypeOf((*command.Interface)(nil)).Elem()
	obj := pkg.Types.Scope().Lookup(it.Name()).Type().Underlying().(*types.Interface)

	// Load command metadata corresponding to each interface method.
	var commands []*Command
	loader := fieldLoader{make(map[types.Object]*Field)}
	for i := 0; i < obj.NumMethods(); i++ {
		m := obj.Method(i)
		c, err := loader.loadMethod(pkg, m)
		if err != nil {
			return nil, nil, fmt.Errorf("loading %s: %v", m.Name(), err)
		}
		commands = append(commands, c)
	}
	return pkg, commands, nil
}

// fieldLoader loads field information, memoizing results to prevent infinite
// recursion.
type fieldLoader struct {
	loaded map[types.Object]*Field
}

var universeError = types.Universe.Lookup("error").Type()

func (l *fieldLoader) loadMethod(pkg *packages.Package, m *types.Func) (*Command, error) {
	node, err := findField(pkg, m.Pos())
	if err != nil {
		return nil, err
	}
	title, doc := splitDoc(node.Doc.Text())
	c := &Command{
		MethodName: m.Name(),
		Name:       lspName(m.Name()),
		Doc:        doc,
		Title:      title,
	}
	sig := m.Type().Underlying().(*types.Signature)
	rlen := sig.Results().Len()
	if rlen > 2 || rlen == 0 {
		return nil, fmt.Errorf("must have 1 or 2 returns, got %d", rlen)
	}
	finalResult := sig.Results().At(rlen - 1)
	if !types.Identical(finalResult.Type(), universeError) {
		return nil, fmt.Errorf("final return must be error")
	}
	if rlen == 2 {
		obj := sig.Results().At(0)
		c.Result, err = l.loadField(pkg, obj, "", "")
		if err != nil {
			return nil, err
		}
	}
	for i := 0; i < sig.Params().Len(); i++ {
		obj := sig.Params().At(i)
		fld, err := l.loadField(pkg, obj, "", "")
		if err != nil {
			return nil, err
		}
		if i == 0 {
			// Lazy check that the first argument is a context. We could relax this,
			// but then the generated code gets more complicated.
			if named, ok := fld.Type.(*types.Named); !ok || named.Obj().Name() != "Context" || named.Obj().Pkg().Path() != "context" {
				return nil, fmt.Errorf("first method parameter must be context.Context")
			}
			// Skip the context argument, as it is implied.
			continue
		}
		c.Args = append(c.Args, fld)
	}
	return c, nil
}

func (l *fieldLoader) loadField(pkg *packages.Package, obj *types.Var, doc, tag string) (*Field, error) {
	if existing, ok := l.loaded[obj]; ok {
		return existing, nil
	}
	fld := &Field{
		Name:    obj.Name(),
		Doc:     strings.TrimSpace(doc),
		Type:    obj.Type(),
		JSONTag: reflect.StructTag(tag).Get("json"),
	}
	under := fld.Type.Underlying()
	// Quick-and-dirty handling for various underlying types.
	switch p := under.(type) {
	case *types.Pointer:
		under = p.Elem().Underlying()
	case *types.Array:
		under = p.Elem().Underlying()
		fld.FieldMod = fmt.Sprintf("[%d]", p.Len())
	case *types.Slice:
		under = p.Elem().Underlying()
		fld.FieldMod = "[]"
	}

	if s, ok := under.(*types.Struct); ok {
		for i := 0; i < s.NumFields(); i++ {
			obj2 := s.Field(i)
			pkg2 := pkg
			if obj2.Pkg() != pkg2.Types {
				pkg2, ok = pkg.Imports[obj2.Pkg().Path()]
				if !ok {
					return nil, fmt.Errorf("missing import for %q: %q", pkg.ID, obj2.Pkg().Path())
				}
			}
			node, err := findField(pkg2, obj2.Pos())
			if err != nil {
				return nil, err
			}
			tag := s.Tag(i)
			structField, err := l.loadField(pkg2, obj2, node.Doc.Text(), tag)
			if err != nil {
				return nil, err
			}
			fld.Fields = append(fld.Fields, structField)
		}
	}
	return fld, nil
}

// splitDoc parses a command doc string to separate the title from normal
// documentation.
//
// The doc comment should be of the form: "MethodName: Title\nDocumentation"
func splitDoc(text string) (title, doc string) {
	docParts := strings.SplitN(text, "\n", 2)
	titleParts := strings.SplitN(docParts[0], ":", 2)
	if len(titleParts) > 1 {
		title = strings.TrimSpace(titleParts[1])
	}
	if len(docParts) > 1 {
		doc = strings.TrimSpace(docParts[1])
	}
	return title, doc
}

// lspName returns the normalized command name to use in the LSP.
func lspName(methodName string) string {
	words := splitCamel(methodName)
	for i := range words {
		words[i] = strings.ToLower(words[i])
	}
	return strings.Join(words, "_")
}

// splitCamel splits s into words, according to camel-case word boundaries.
// Initialisms are grouped as a single word.
//
// For example:
//
//	"RunTests" -> []string{"Run", "Tests"}
//	"GCDetails" -> []string{"GC", "Details"}
func splitCamel(s string) []string {
	var words []string
	for len(s) > 0 {
		last := strings.LastIndexFunc(s, unicode.IsUpper)
		if last < 0 {
			last = 0
		}
		if last == len(s)-1 {
			// Group initialisms as a single word.
			last = 1 + strings.LastIndexFunc(s[:last], func(r rune) bool { return !unicode.IsUpper(r) })
		}
		words = append(words, s[last:])
		s = s[:last]
	}
	for i := 0; i < len(words)/2; i++ {
		j := len(words) - i - 1
		words[i], words[j] = words[j], words[i]
	}
	return words
}

// findField finds the struct field or interface method positioned at pos,
// within the AST.
func findField(pkg *packages.Package, pos token.Pos) (*ast.Field, error) {
	fset := pkg.Fset
	var file *ast.File
	for _, f := range pkg.Syntax {
		if fset.Position(f.Pos()).Filename == fset.Position(pos).Filename {
			file = f
			break
		}
	}
	if file == nil {
		return nil, fmt.Errorf("no file for pos %v", pos)
	}
	path, _ := astutil.PathEnclosingInterval(file, pos, pos)
	// This is fragile, but in the cases we care about, the field will be in
	// path[1].
	return path[1].(*ast.Field), nil
}
