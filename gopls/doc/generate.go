// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command generate creates API (settings, etc) documentation in JSON and
// Markdown for machine and human consumption.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"time"

	"github.com/sanity-io/litter"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/source"
)

func main() {
	if _, err := doMain("..", true); err != nil {
		fmt.Fprintf(os.Stderr, "Generation failed: %v\n", err)
		os.Exit(1)
	}
}

func doMain(baseDir string, write bool) (bool, error) {
	api, err := loadAPI()
	if err != nil {
		return false, err
	}

	if ok, err := rewriteFile(filepath.Join(baseDir, "internal/lsp/source/api_json.go"), api, write, rewriteAPI); !ok || err != nil {
		return ok, err
	}
	if ok, err := rewriteFile(filepath.Join(baseDir, "gopls/doc/settings.md"), api, write, rewriteSettings); !ok || err != nil {
		return ok, err
	}
	if ok, err := rewriteFile(filepath.Join(baseDir, "gopls/doc/commands.md"), api, write, rewriteCommands); !ok || err != nil {
		return ok, err
	}

	return true, nil
}

func loadAPI() (*source.APIJSON, error) {
	pkgs, err := packages.Load(
		&packages.Config{
			Mode: packages.NeedTypes | packages.NeedTypesInfo | packages.NeedSyntax | packages.NeedDeps,
		},
		"golang.org/x/tools/internal/lsp/source",
	)
	if err != nil {
		return nil, err
	}
	pkg := pkgs[0]

	api := &source.APIJSON{
		Options: map[string][]*source.OptionJSON{},
	}
	defaults := source.DefaultOptions()
	for _, cat := range []reflect.Value{
		reflect.ValueOf(defaults.DebuggingOptions),
		reflect.ValueOf(defaults.UserOptions),
		reflect.ValueOf(defaults.ExperimentalOptions),
	} {
		opts, err := loadOptions(cat, pkg)
		if err != nil {
			return nil, err
		}
		catName := strings.TrimSuffix(cat.Type().Name(), "Options")
		api.Options[catName] = opts
	}

	api.Commands, err = loadCommands(pkg)
	if err != nil {
		return nil, err
	}
	api.Lenses = loadLenses(api.Commands)

	// Transform the internal command name to the external command name.
	for _, c := range api.Commands {
		c.Command = source.CommandPrefix + c.Command
	}
	return api, nil
}

func loadOptions(category reflect.Value, pkg *packages.Package) ([]*source.OptionJSON, error) {
	// Find the type information and ast.File corresponding to the category.
	optsType := pkg.Types.Scope().Lookup(category.Type().Name())
	if optsType == nil {
		return nil, fmt.Errorf("could not find %v in scope %v", category.Type().Name(), pkg.Types.Scope())
	}

	file, err := fileForPos(pkg, optsType.Pos())
	if err != nil {
		return nil, err
	}

	enums, err := loadEnums(pkg)
	if err != nil {
		return nil, err
	}

	var opts []*source.OptionJSON
	optsStruct := optsType.Type().Underlying().(*types.Struct)
	for i := 0; i < optsStruct.NumFields(); i++ {
		// The types field gives us the type.
		typesField := optsStruct.Field(i)
		path, _ := astutil.PathEnclosingInterval(file, typesField.Pos(), typesField.Pos())
		if len(path) < 2 {
			return nil, fmt.Errorf("could not find AST node for field %v", typesField)
		}
		// The AST field gives us the doc.
		astField, ok := path[1].(*ast.Field)
		if !ok {
			return nil, fmt.Errorf("unexpected AST path %v", path)
		}

		// The reflect field gives us the default value.
		reflectField := category.FieldByName(typesField.Name())
		if !reflectField.IsValid() {
			return nil, fmt.Errorf("could not find reflect field for %v", typesField.Name())
		}

		// Format the default value. VSCode exposes settings as JSON, so showing them as JSON is reasonable.
		def := reflectField.Interface()
		// Durations marshal as nanoseconds, but we want the stringy versions, e.g. "100ms".
		if t, ok := def.(time.Duration); ok {
			def = t.String()
		}
		defBytes, err := json.Marshal(def)
		if err != nil {
			return nil, err
		}

		// Nil values format as "null" so print them as hardcoded empty values.
		switch reflectField.Type().Kind() {
		case reflect.Map:
			if reflectField.IsNil() {
				defBytes = []byte("{}")
			}
		case reflect.Slice:
			if reflectField.IsNil() {
				defBytes = []byte("[]")
			}
		}

		typ := typesField.Type().String()
		if _, ok := enums[typesField.Type()]; ok {
			typ = "enum"
		}

		opts = append(opts, &source.OptionJSON{
			Name:       lowerFirst(typesField.Name()),
			Type:       typ,
			Doc:        lowerFirst(astField.Doc.Text()),
			Default:    string(defBytes),
			EnumValues: enums[typesField.Type()],
		})
	}
	return opts, nil
}

func loadEnums(pkg *packages.Package) (map[types.Type][]source.EnumValue, error) {
	enums := map[types.Type][]source.EnumValue{}
	for _, name := range pkg.Types.Scope().Names() {
		obj := pkg.Types.Scope().Lookup(name)
		cnst, ok := obj.(*types.Const)
		if !ok {
			continue
		}
		f, err := fileForPos(pkg, cnst.Pos())
		if err != nil {
			return nil, fmt.Errorf("finding file for %q: %v", cnst.Name(), err)
		}
		path, _ := astutil.PathEnclosingInterval(f, cnst.Pos(), cnst.Pos())
		spec := path[1].(*ast.ValueSpec)
		value := cnst.Val().ExactString()
		doc := valueDoc(cnst.Name(), value, spec.Doc.Text())
		v := source.EnumValue{
			Value: value,
			Doc:   doc,
		}
		enums[obj.Type()] = append(enums[obj.Type()], v)
	}
	return enums, nil
}

// valueDoc transforms a docstring documenting an constant identifier to a
// docstring documenting its value.
//
// If doc is of the form "Foo is a bar", it returns '`"fooValue"` is a bar'. If
// doc is non-standard ("this value is a bar"), it returns '`"fooValue"`: this
// value is a bar'.
func valueDoc(name, value, doc string) string {
	if doc == "" {
		return ""
	}
	if strings.HasPrefix(doc, name) {
		// docstring in standard form. Replace the subject with value.
		return fmt.Sprintf("`%s`%s", value, doc[len(name):])
	}
	return fmt.Sprintf("`%s`: %s", value, doc)
}

func loadCommands(pkg *packages.Package) ([]*source.CommandJSON, error) {
	// The code that defines commands is much more complicated than the
	// code that defines options, so reading comments for the Doc is very
	// fragile. If this causes problems, we should switch to a dynamic
	// approach and put the doc in the Commands struct rather than reading
	// from the source code.

	// Find the Commands slice.
	typesSlice := pkg.Types.Scope().Lookup("Commands")
	f, err := fileForPos(pkg, typesSlice.Pos())
	if err != nil {
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(f, typesSlice.Pos(), typesSlice.Pos())
	vspec := path[1].(*ast.ValueSpec)
	var astSlice *ast.CompositeLit
	for i, name := range vspec.Names {
		if name.Name == "Commands" {
			astSlice = vspec.Values[i].(*ast.CompositeLit)
		}
	}

	var commands []*source.CommandJSON

	// Parse the objects it contains.
	for _, elt := range astSlice.Elts {
		// Find the composite literal of the Command.
		typesCommand := pkg.TypesInfo.ObjectOf(elt.(*ast.Ident))
		path, _ := astutil.PathEnclosingInterval(f, typesCommand.Pos(), typesCommand.Pos())
		vspec := path[1].(*ast.ValueSpec)

		var astCommand ast.Expr
		for i, name := range vspec.Names {
			if name.Name == typesCommand.Name() {
				astCommand = vspec.Values[i]
			}
		}

		// Read the Name and Title fields of the literal.
		var name, title string
		ast.Inspect(astCommand, func(n ast.Node) bool {
			kv, ok := n.(*ast.KeyValueExpr)
			if ok {
				k := kv.Key.(*ast.Ident).Name
				switch k {
				case "Name":
					name = strings.Trim(kv.Value.(*ast.BasicLit).Value, `"`)
				case "Title":
					title = strings.Trim(kv.Value.(*ast.BasicLit).Value, `"`)
				}
			}
			return true
		})

		if title == "" {
			title = name
		}

		// Conventionally, the doc starts with the name of the variable.
		// Replace it with the name of the command.
		doc := vspec.Doc.Text()
		doc = strings.Replace(doc, typesCommand.Name(), name, 1)

		commands = append(commands, &source.CommandJSON{
			Command: name,
			Title:   title,
			Doc:     doc,
		})
	}
	return commands, nil
}

func loadLenses(commands []*source.CommandJSON) []*source.LensJSON {
	lensNames := map[string]struct{}{}
	for k := range source.LensFuncs() {
		lensNames[k] = struct{}{}
	}
	for k := range mod.LensFuncs() {
		lensNames[k] = struct{}{}
	}

	var lenses []*source.LensJSON

	for _, cmd := range commands {
		if _, ok := lensNames[cmd.Command]; ok {
			lenses = append(lenses, &source.LensJSON{
				Lens:  cmd.Command,
				Title: cmd.Title,
				Doc:   cmd.Doc,
			})
		}
	}
	return lenses
}

func lowerFirst(x string) string {
	if x == "" {
		return x
	}
	return strings.ToLower(x[:1]) + x[1:]
}

func fileForPos(pkg *packages.Package, pos token.Pos) (*ast.File, error) {
	fset := pkg.Fset
	for _, f := range pkg.Syntax {
		if fset.Position(f.Pos()).Filename == fset.Position(pos).Filename {
			return f, nil
		}
	}
	return nil, fmt.Errorf("no file for pos %v", pos)
}

func rewriteFile(file string, api *source.APIJSON, write bool, rewrite func([]byte, *source.APIJSON) ([]byte, error)) (bool, error) {
	old, err := ioutil.ReadFile(file)
	if err != nil {
		return false, err
	}

	new, err := rewrite(old, api)
	if err != nil {
		return false, fmt.Errorf("rewriting %q: %v", file, err)
	}

	if !write {
		return bytes.Equal(old, new), nil
	}

	if err := ioutil.WriteFile(file, new, 0); err != nil {
		return false, err
	}

	return true, nil
}

func rewriteAPI(input []byte, api *source.APIJSON) ([]byte, error) {
	buf := bytes.NewBuffer(nil)
	apiStr := litter.Options{
		HomePackage: "source",
	}.Sdump(api)
	// Massive hack: filter out redundant types from the composite literal.
	apiStr = strings.ReplaceAll(apiStr, "&OptionJSON", "")
	apiStr = strings.ReplaceAll(apiStr, ": []*OptionJSON", ":")
	apiStr = strings.ReplaceAll(apiStr, "&CommandJSON", "")
	apiStr = strings.ReplaceAll(apiStr, "&LensJSON", "")
	apiStr = strings.ReplaceAll(apiStr, "  EnumValue{", "{")
	apiBytes, err := format.Source([]byte(apiStr))
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(buf, "// Code generated by \"golang.org/x/tools/gopls/doc/generate\"; DO NOT EDIT.\n\npackage source\n\nvar GeneratedAPIJSON = %s\n", apiBytes)
	return buf.Bytes(), nil
}

var parBreakRE = regexp.MustCompile("\n{2,}")

func rewriteSettings(doc []byte, api *source.APIJSON) ([]byte, error) {
	result := doc
	for category, opts := range api.Options {
		section := bytes.NewBuffer(nil)
		for _, opt := range opts {
			var enumValues strings.Builder
			if len(opt.EnumValues) > 0 {
				enumValues.WriteString("Must be one of:\n\n")
				for _, val := range opt.EnumValues {
					if val.Doc != "" {
						// Don't break the list item by starting a new paragraph.
						unbroken := parBreakRE.ReplaceAllString(val.Doc, "\\\n")
						fmt.Fprintf(&enumValues, " * %s\n", unbroken)
					} else {
						fmt.Fprintf(&enumValues, " * `%s`\n", val.Value)
					}
				}
			}
			fmt.Fprintf(section, "### **%v** *%v*\n%v%v\n\nDefault: `%v`.\n", opt.Name, opt.Type, opt.Doc, enumValues.String(), opt.Default)
		}
		var err error
		result, err = replaceSection(result, category, section.Bytes())
		if err != nil {
			return nil, err
		}
	}

	section := bytes.NewBuffer(nil)
	for _, lens := range api.Lenses {
		fmt.Fprintf(section, "### **%v**\nIdentifier: `%v`\n\n%v\n\n", lens.Title, lens.Lens, lens.Doc)
	}
	return replaceSection(result, "Lenses", section.Bytes())
}

func rewriteCommands(doc []byte, api *source.APIJSON) ([]byte, error) {
	section := bytes.NewBuffer(nil)
	for _, command := range api.Commands {
		fmt.Fprintf(section, "### **%v**\nIdentifier: `%v`\n\n%v\n\n", command.Title, command.Command, command.Doc)
	}
	return replaceSection(doc, "Commands", section.Bytes())
}

func replaceSection(doc []byte, sectionName string, replacement []byte) ([]byte, error) {
	re := regexp.MustCompile(fmt.Sprintf(`(?s)<!-- BEGIN %v.* -->\n(.*?)<!-- END %v.* -->`, sectionName, sectionName))
	idx := re.FindSubmatchIndex(doc)
	if idx == nil {
		return nil, fmt.Errorf("could not find section %q", sectionName)
	}
	result := append([]byte(nil), doc[:idx[2]]...)
	result = append(result, replacement...)
	result = append(result, doc[idx[3]:]...)
	return result, nil
}
