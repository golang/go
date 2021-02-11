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
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/sanity-io/litter"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/command/commandmeta"
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
	if ok, err := rewriteFile(filepath.Join(baseDir, "gopls/doc/analyzers.md"), api, write, rewriteAnalyzers); !ok || err != nil {
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

	api.Commands, err = loadCommands(pkg)
	if err != nil {
		return nil, err
	}
	api.Lenses = loadLenses(api.Commands)

	// Transform the internal command name to the external command name.
	for _, c := range api.Commands {
		c.Command = command.ID(c.Command)
	}
	for _, m := range []map[string]*source.Analyzer{
		defaults.DefaultAnalyzers,
		defaults.TypeErrorAnalyzers,
		defaults.ConvenienceAnalyzers,
		// Don't yet add staticcheck analyzers.
	} {
		api.Analyzers = append(api.Analyzers, loadAnalyzers(m)...)
	}
	for _, category := range []reflect.Value{
		reflect.ValueOf(defaults.UserOptions),
	} {
		// Find the type information and ast.File corresponding to the category.
		optsType := pkg.Types.Scope().Lookup(category.Type().Name())
		if optsType == nil {
			return nil, fmt.Errorf("could not find %v in scope %v", category.Type().Name(), pkg.Types.Scope())
		}
		opts, err := loadOptions(category, optsType, pkg, "")
		if err != nil {
			return nil, err
		}
		catName := strings.TrimSuffix(category.Type().Name(), "Options")
		api.Options[catName] = opts

		// Hardcode the expected values for the analyses and code lenses
		// settings, since their keys are not enums.
		for _, opt := range opts {
			switch opt.Name {
			case "analyses":
				for _, a := range api.Analyzers {
					opt.EnumKeys.Keys = append(opt.EnumKeys.Keys, source.EnumKey{
						Name:    fmt.Sprintf("%q", a.Name),
						Doc:     a.Doc,
						Default: strconv.FormatBool(a.Default),
					})
				}
			case "codelenses":
				// Hack: Lenses don't set default values, and we don't want to
				// pass in the list of expected lenses to loadOptions. Instead,
				// format the defaults using reflection here. The hackiest part
				// is reversing lowercasing of the field name.
				reflectField := category.FieldByName(upperFirst(opt.Name))
				for _, l := range api.Lenses {
					def, err := formatDefaultFromEnumBoolMap(reflectField, l.Lens)
					if err != nil {
						return nil, err
					}
					opt.EnumKeys.Keys = append(opt.EnumKeys.Keys, source.EnumKey{
						Name:    fmt.Sprintf("%q", l.Lens),
						Doc:     l.Doc,
						Default: def,
					})
				}
			}
		}
	}
	return api, nil
}

func loadOptions(category reflect.Value, optsType types.Object, pkg *packages.Package, hierarchy string) ([]*source.OptionJSON, error) {
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

		// If the field name ends with "Options", assume it is a struct with
		// additional options and process it recursively.
		if h := strings.TrimSuffix(typesField.Name(), "Options"); h != typesField.Name() {
			// Keep track of the parent structs.
			if hierarchy != "" {
				h = hierarchy + "." + h
			}
			options, err := loadOptions(category, typesField, pkg, strings.ToLower(h))
			if err != nil {
				return nil, err
			}
			opts = append(opts, options...)
			continue
		}
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

		def, err := formatDefault(reflectField)
		if err != nil {
			return nil, err
		}

		typ := typesField.Type().String()
		if _, ok := enums[typesField.Type()]; ok {
			typ = "enum"
		}
		name := lowerFirst(typesField.Name())

		var enumKeys source.EnumKeys
		if m, ok := typesField.Type().(*types.Map); ok {
			e, ok := enums[m.Key()]
			if ok {
				typ = strings.Replace(typ, m.Key().String(), m.Key().Underlying().String(), 1)
			}
			keys, err := collectEnumKeys(name, m, reflectField, e)
			if err != nil {
				return nil, err
			}
			if keys != nil {
				enumKeys = *keys
			}
		}

		// Get the status of the field by checking its struct tags.
		reflectStructField, ok := category.Type().FieldByName(typesField.Name())
		if !ok {
			return nil, fmt.Errorf("no struct field for %s", typesField.Name())
		}
		status := reflectStructField.Tag.Get("status")

		opts = append(opts, &source.OptionJSON{
			Name:       name,
			Type:       typ,
			Doc:        lowerFirst(astField.Doc.Text()),
			Default:    def,
			EnumKeys:   enumKeys,
			EnumValues: enums[typesField.Type()],
			Status:     status,
			Hierarchy:  hierarchy,
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

func collectEnumKeys(name string, m *types.Map, reflectField reflect.Value, enumValues []source.EnumValue) (*source.EnumKeys, error) {
	// Make sure the value type gets set for analyses and codelenses
	// too.
	if len(enumValues) == 0 && !hardcodedEnumKeys(name) {
		return nil, nil
	}
	keys := &source.EnumKeys{
		ValueType: m.Elem().String(),
	}
	// We can get default values for enum -> bool maps.
	var isEnumBoolMap bool
	if basic, ok := m.Elem().(*types.Basic); ok && basic.Kind() == types.Bool {
		isEnumBoolMap = true
	}
	for _, v := range enumValues {
		var def string
		if isEnumBoolMap {
			var err error
			def, err = formatDefaultFromEnumBoolMap(reflectField, v.Value)
			if err != nil {
				return nil, err
			}
		}
		keys.Keys = append(keys.Keys, source.EnumKey{
			Name:    v.Value,
			Doc:     v.Doc,
			Default: def,
		})
	}
	return keys, nil
}

func formatDefaultFromEnumBoolMap(reflectMap reflect.Value, enumKey string) (string, error) {
	if reflectMap.Kind() != reflect.Map {
		return "", nil
	}
	name := enumKey
	if unquoted, err := strconv.Unquote(name); err == nil {
		name = unquoted
	}
	for _, e := range reflectMap.MapKeys() {
		if e.String() == name {
			value := reflectMap.MapIndex(e)
			if value.Type().Kind() == reflect.Bool {
				return formatDefault(value)
			}
		}
	}
	// Assume that if the value isn't mentioned in the map, it defaults to
	// the default value, false.
	return formatDefault(reflect.ValueOf(false))
}

// formatDefault formats the default value into a JSON-like string.
// VS Code exposes settings as JSON, so showing them as JSON is reasonable.
// TODO(rstambler): Reconsider this approach, as the VS Code Go generator now
// marshals to JSON.
func formatDefault(reflectField reflect.Value) (string, error) {
	def := reflectField.Interface()

	// Durations marshal as nanoseconds, but we want the stringy versions,
	// e.g. "100ms".
	if t, ok := def.(time.Duration); ok {
		def = t.String()
	}
	defBytes, err := json.Marshal(def)
	if err != nil {
		return "", err
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
	return string(defBytes), err
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

	var commands []*source.CommandJSON

	_, cmds, err := commandmeta.Load()
	if err != nil {
		return nil, err
	}
	// Parse the objects it contains.
	for _, cmd := range cmds {
		cmdjson := &source.CommandJSON{
			Command: cmd.Name,
			Title:   cmd.Title,
			Doc:     cmd.Doc,
			ArgDoc:  argsDoc(cmd.Args),
		}
		if cmd.Result != nil {
			cmdjson.ResultDoc = typeDoc(cmd.Result, 0)
		}
		commands = append(commands, cmdjson)
	}
	return commands, nil
}

func argsDoc(args []*commandmeta.Field) string {
	var b strings.Builder
	for i, arg := range args {
		b.WriteString(typeDoc(arg, 0))
		if i != len(args)-1 {
			b.WriteString(",\n")
		}
	}
	return b.String()
}

func typeDoc(arg *commandmeta.Field, level int) string {
	// Max level to expand struct fields.
	const maxLevel = 3
	if len(arg.Fields) > 0 {
		if level < maxLevel {
			return arg.FieldMod + structDoc(arg.Fields, level)
		}
		return "{ ... }"
	}
	under := arg.Type.Underlying()
	switch u := under.(type) {
	case *types.Slice:
		return fmt.Sprintf("[]%s", u.Elem().Underlying().String())
	}
	return types.TypeString(under, nil)
}

func structDoc(fields []*commandmeta.Field, level int) string {
	var b strings.Builder
	b.WriteString("{\n")
	indent := strings.Repeat("\t", level)
	for _, fld := range fields {
		if fld.Doc != "" && level == 0 {
			doclines := strings.Split(fld.Doc, "\n")
			for _, line := range doclines {
				fmt.Fprintf(&b, "%s\t// %s\n", indent, line)
			}
		}
		tag := fld.JSONTag
		if tag == "" {
			tag = fld.Name
		}
		fmt.Fprintf(&b, "%s\t%q: %s,\n", indent, tag, typeDoc(fld, level+1))
	}
	fmt.Fprintf(&b, "%s}", indent)
	return b.String()
}

func loadLenses(commands []*source.CommandJSON) []*source.LensJSON {
	all := map[command.Command]struct{}{}
	for k := range source.LensFuncs() {
		all[k] = struct{}{}
	}
	for k := range mod.LensFuncs() {
		if _, ok := all[k]; ok {
			panic(fmt.Sprintf("duplicate lens %q", string(k)))
		}
		all[k] = struct{}{}
	}

	var lenses []*source.LensJSON

	for _, cmd := range commands {
		if _, ok := all[command.Command(cmd.Command)]; ok {
			lenses = append(lenses, &source.LensJSON{
				Lens:  cmd.Command,
				Title: cmd.Title,
				Doc:   cmd.Doc,
			})
		}
	}
	return lenses
}

func loadAnalyzers(m map[string]*source.Analyzer) []*source.AnalyzerJSON {
	var sorted []string
	for _, a := range m {
		sorted = append(sorted, a.Analyzer.Name)
	}
	sort.Strings(sorted)
	var json []*source.AnalyzerJSON
	for _, name := range sorted {
		a := m[name]
		json = append(json, &source.AnalyzerJSON{
			Name:    a.Analyzer.Name,
			Doc:     a.Analyzer.Doc,
			Default: a.Enabled,
		})
	}
	return json
}

func lowerFirst(x string) string {
	if x == "" {
		return x
	}
	return strings.ToLower(x[:1]) + x[1:]
}

func upperFirst(x string) string {
	if x == "" {
		return x
	}
	return strings.ToUpper(x[:1]) + x[1:]
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

func rewriteAPI(_ []byte, api *source.APIJSON) ([]byte, error) {
	buf := bytes.NewBuffer(nil)
	apiStr := litter.Options{
		HomePackage: "source",
	}.Sdump(api)
	// Massive hack: filter out redundant types from the composite literal.
	apiStr = strings.ReplaceAll(apiStr, "&OptionJSON", "")
	apiStr = strings.ReplaceAll(apiStr, ": []*OptionJSON", ":")
	apiStr = strings.ReplaceAll(apiStr, "&CommandJSON", "")
	apiStr = strings.ReplaceAll(apiStr, "&LensJSON", "")
	apiStr = strings.ReplaceAll(apiStr, "&AnalyzerJSON", "")
	apiStr = strings.ReplaceAll(apiStr, "  EnumValue{", "{")
	apiStr = strings.ReplaceAll(apiStr, "  EnumKey{", "{")
	apiBytes, err := format.Source([]byte(apiStr))
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(buf, "// Code generated by \"golang.org/x/tools/gopls/doc/generate\"; DO NOT EDIT.\n\npackage source\n\nvar GeneratedAPIJSON = %s\n", apiBytes)
	return buf.Bytes(), nil
}

var parBreakRE = regexp.MustCompile("\n{2,}")

type optionsGroup struct {
	title   string
	final   string
	level   int
	options []*source.OptionJSON
}

func rewriteSettings(doc []byte, api *source.APIJSON) ([]byte, error) {
	result := doc
	for category, opts := range api.Options {
		groups := collectGroups(opts)

		// First, print a table of contents.
		section := bytes.NewBuffer(nil)
		fmt.Fprintln(section, "")
		for _, h := range groups {
			writeBullet(section, h.final, h.level)
		}
		fmt.Fprintln(section, "")

		// Currently, the settings document has a title and a subtitle, so
		// start at level 3 for a header beginning with "###".
		baseLevel := 3
		for _, h := range groups {
			level := baseLevel + h.level
			writeTitle(section, h.final, level)
			for _, opt := range h.options {
				header := strMultiply("#", level+1)
				fmt.Fprintf(section, "%s **%v** *%v*\n\n", header, opt.Name, opt.Type)
				writeStatus(section, opt.Status)
				enumValues := collectEnums(opt)
				fmt.Fprintf(section, "%v%v\nDefault: `%v`.\n\n", opt.Doc, enumValues, opt.Default)
			}
		}
		var err error
		result, err = replaceSection(result, category, section.Bytes())
		if err != nil {
			return nil, err
		}
	}

	section := bytes.NewBuffer(nil)
	for _, lens := range api.Lenses {
		fmt.Fprintf(section, "### **%v**\n\nIdentifier: `%v`\n\n%v\n", lens.Title, lens.Lens, lens.Doc)
	}
	return replaceSection(result, "Lenses", section.Bytes())
}

func collectGroups(opts []*source.OptionJSON) []optionsGroup {
	optsByHierarchy := map[string][]*source.OptionJSON{}
	for _, opt := range opts {
		optsByHierarchy[opt.Hierarchy] = append(optsByHierarchy[opt.Hierarchy], opt)
	}

	// As a hack, assume that uncategorized items are less important to
	// users and force the empty string to the end of the list.
	var containsEmpty bool
	var sorted []string
	for h := range optsByHierarchy {
		if h == "" {
			containsEmpty = true
			continue
		}
		sorted = append(sorted, h)
	}
	sort.Strings(sorted)
	if containsEmpty {
		sorted = append(sorted, "")
	}
	var groups []optionsGroup
	baseLevel := 0
	for _, h := range sorted {
		split := strings.SplitAfter(h, ".")
		last := split[len(split)-1]
		// Hack to capitalize all of UI.
		if last == "ui" {
			last = "UI"
		}
		// A hierarchy may look like "ui.formatting". If "ui" has no
		// options of its own, it may not be added to the map, but it
		// still needs a heading.
		components := strings.Split(h, ".")
		for i := 1; i < len(components); i++ {
			parent := strings.Join(components[0:i], ".")
			if _, ok := optsByHierarchy[parent]; !ok {
				groups = append(groups, optionsGroup{
					title: parent,
					final: last,
					level: baseLevel + i,
				})
			}
		}
		groups = append(groups, optionsGroup{
			title:   h,
			final:   last,
			level:   baseLevel + strings.Count(h, "."),
			options: optsByHierarchy[h],
		})
	}
	return groups
}

func collectEnums(opt *source.OptionJSON) string {
	var b strings.Builder
	write := func(name, doc string, index, len int) {
		if doc != "" {
			unbroken := parBreakRE.ReplaceAllString(doc, "\\\n")
			fmt.Fprintf(&b, "* %s", unbroken)
		} else {
			fmt.Fprintf(&b, "* `%s`", name)
		}
		if index < len-1 {
			fmt.Fprint(&b, "\n")
		}
	}
	if len(opt.EnumValues) > 0 && opt.Type == "enum" {
		b.WriteString("\nMust be one of:\n\n")
		for i, val := range opt.EnumValues {
			write(val.Value, val.Doc, i, len(opt.EnumValues))
		}
	} else if len(opt.EnumKeys.Keys) > 0 && shouldShowEnumKeysInSettings(opt.Name) {
		b.WriteString("\nCan contain any of:\n\n")
		for i, val := range opt.EnumKeys.Keys {
			write(val.Name, val.Doc, i, len(opt.EnumKeys.Keys))
		}
	}
	return b.String()
}

func shouldShowEnumKeysInSettings(name string) bool {
	// Both of these fields have too many possible options to print.
	return !hardcodedEnumKeys(name)
}

func hardcodedEnumKeys(name string) bool {
	return name == "analyses" || name == "codelenses"
}

func writeBullet(w io.Writer, title string, level int) {
	if title == "" {
		return
	}
	// Capitalize the first letter of each title.
	prefix := strMultiply("  ", level)
	fmt.Fprintf(w, "%s* [%s](#%s)\n", prefix, capitalize(title), strings.ToLower(title))
}

func writeTitle(w io.Writer, title string, level int) {
	if title == "" {
		return
	}
	// Capitalize the first letter of each title.
	fmt.Fprintf(w, "%s %s\n\n", strMultiply("#", level), capitalize(title))
}

func writeStatus(section io.Writer, status string) {
	switch status {
	case "":
	case "advanced":
		fmt.Fprint(section, "**This is an advanced setting and should not be configured by most `gopls` users.**\n\n")
	case "debug":
		fmt.Fprint(section, "**This setting is for debugging purposes only.**\n\n")
	case "experimental":
		fmt.Fprint(section, "**This setting is experimental and may be deleted.**\n\n")
	default:
		fmt.Fprintf(section, "**Status: %s.**\n\n", status)
	}
}

func capitalize(s string) string {
	return string(unicode.ToUpper(rune(s[0]))) + s[1:]
}

func strMultiply(str string, count int) string {
	var result string
	for i := 0; i < count; i++ {
		result += string(str)
	}
	return result
}

func rewriteCommands(doc []byte, api *source.APIJSON) ([]byte, error) {
	section := bytes.NewBuffer(nil)
	for _, command := range api.Commands {
		fmt.Fprintf(section, "### **%v**\nIdentifier: `%v`\n\n%v\n\n", command.Title, command.Command, command.Doc)
		if command.ArgDoc != "" {
			fmt.Fprintf(section, "Args:\n\n```\n%s\n```\n\n", command.ArgDoc)
		}
		if command.ResultDoc != "" {
			fmt.Fprintf(section, "Result:\n\n```\n%s\n```\n\n", command.ResultDoc)
		}
	}
	return replaceSection(doc, "Commands", section.Bytes())
}

func rewriteAnalyzers(doc []byte, api *source.APIJSON) ([]byte, error) {
	section := bytes.NewBuffer(nil)
	for _, analyzer := range api.Analyzers {
		fmt.Fprintf(section, "## **%v**\n\n", analyzer.Name)
		fmt.Fprintf(section, "%s\n\n", analyzer.Doc)
		switch analyzer.Default {
		case true:
			fmt.Fprintf(section, "**Enabled by default.**\n\n")
		case false:
			fmt.Fprintf(section, "**Disabled by default. Enable it by setting `\"analyses\": {\"%s\": true}`.**\n\n", analyzer.Name)
		}
	}
	return replaceSection(doc, "Analyzers", section.Bytes())
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
