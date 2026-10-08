// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"errors"
	"go/ast"
	"go/token"
	"go/types"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"golang.org/x/tools/go/packages"
	"gopkg.in/yaml.v3"

	"simd/archsimd/_gen/gentools"
)

// buildConfigs are the configurations we report API coverage for. The API is spread
// across files selected by build constraints, and the only authority on which
// files belong to which configuration is the go command -- so rather than
// interpreting //go:build lines here, we ask go/packages once per configuration
// and use the file set it hands back.
//
// SVE is a target in SPEC-TRANSITION.md's sense but not a GOARCH: its files are
// built for arm64 like NEON's. It separates out along the generator dimension
// instead, because simdgen records its -arch in the generated-by header.
var buildConfigs = []struct{ name, goos, goarch string }{
	{"amd64", "linux", "amd64"},
	{"arm64", "linux", "arm64"},
	{"wasm", "js", "wasm"},
	// Package simd falls back to a pure-Go implementation on architectures with
	// no SIMD support. 386 is an arbitrary representative of those.
	{"portable", "linux", "386"},
}

// A decl is one exported function or method declaration in the public API.
type decl struct {
	pkg     string   // "simd" or "archsimd"
	file    string   // path relative to the simd directory
	gen     string   // generator that emitted it, or "hand-written"
	targets []string // names of build configurations it compiles in
	recv    string   // receiver type name, empty for plain functions
	name    string
	in      []param
	out     []param

	doc    string // doc comment text, comment markers stripped
	hasSep bool   // doc contains a bare "//" separator line
}

type param struct{ name, typ string }

// The grains the report counts at, coarsest last. Spec describes an operation
// once, the API declares it once per target, and a name collapses receivers
// too; the gaps between the three are diagnostic (SPEC-TRANSITION.md §1.2).
type (
	// methodKey is the (receiver, name) key spec is unique on.
	methodKey struct{ recv, name string }

	// archMethodKey is a method as it appears in one build configuration.
	//
	// It carries the package because simd and archsimd share their scalable
	// type names.
	archMethodKey struct {
		target string
		pkg    string
		methodKey
	}

	// declKey identifies one declaration in the source, which is what collapses
	// the several configurations it may build in.
	declKey struct {
		file string
		methodKey
	}
)

// A fact is one declaration as seen in one build configuration: the grain of
// the fact table, and the row type every figure in the report pivots over.
type fact struct {
	*decl
	target string
}

func (d *decl) methodKey() methodKey        { return methodKey{d.recv, d.name} }
func (f fact) declKey() declKey             { return declKey{f.file, f.methodKey()} }
func (f fact) archMethodKey() archMethodKey { return archMethodKey{f.target, f.pkg, f.methodKey()} }

// tool strips the invocation from a generator label, so that simdgen's three
// runs count as one tool.
func (f fact) tool() string { t, _, _ := strings.Cut(f.gen, "/"); return t }

// nonSpecOp reports whether name is an operation that will never come from spec:
// they are Go conventions (String, Len) or architecture bridge operations
// (ToArch, FromArch). They are excluded from the fact table, and
// so from every denominator. See SPEC-TRANSITION.md §2.7.
func nonSpecOp(name string) bool {
	return name == "String" || name == "Len" || name == "ToArch" || strings.HasSuffix(name, "FromArch")
}

// loadOverlay builds a packages.Config.Overlay map from the scratch overlay
// directory if one is active.
func loadOverlay(genOpts *gentools.Options) map[string][]byte {
	if genOpts == nil {
		return nil
	}
	dir := genOpts.OverlayDir()
	if dir == "" {
		return nil
	}
	root := filepath.Join(dir, "src")
	overlay := make(map[string][]byte)
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		slashRel := filepath.ToSlash(rel)
		if !strings.HasPrefix(slashRel, "simd/") {
			return nil
		}
		b, err := genOpts.ReadFile(rel)
		if err != nil {
			return nil
		}
		gorootPath := filepath.Join(genOpts.GOROOT, "src", filepath.FromSlash(rel))
		overlay[gorootPath] = b
		return nil
	})
	return overlay
}

func loadAPI(genOpts *gentools.Options, simdDir string) []*decl {
	fset := token.NewFileSet()
	byKey := map[declKey]*decl{}
	overlay := loadOverlay(genOpts)

	for _, bc := range buildConfigs {
		cfg := &packages.Config{
			// Syntax only: we compare declarations across configurations, which
			// type-checking would not help with and would slow down. See
			// SPEC-TRANSITION.md §2.4.
			Mode:    packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles | packages.NeedSyntax,
			Dir:     simdDir,
			Fset:    fset,
			Env:     append(os.Environ(), "GOOS="+bc.goos, "GOARCH="+bc.goarch, "GOEXPERIMENT=simd"),
			Overlay: overlay,
		}
		pkgs, err := packages.Load(cfg, ".", "./archsimd")
		if err != nil {
			log.Fatalf("loading simd packages for %s/%s: %s", bc.goos, bc.goarch, err)
		}
		for _, p := range pkgs {
			for _, e := range p.Errors {
				log.Fatalf("loading %s for %s/%s: %s", p.PkgPath, bc.goos, bc.goarch, e)
			}
			for _, f := range p.Syntax {
				file := fset.Position(f.Pos()).Filename
				if rel, err := filepath.Rel(simdDir, file); err == nil {
					file = filepath.ToSlash(rel)
				}
				gen := generatorOf(f)
				for _, d := range f.Decls {
					fd, ok := d.(*ast.FuncDecl)
					if !ok || !fd.Name.IsExported() {
						continue
					}
					dec := declOf(fd, p.Name, file, gen)
					key := declKey{file, dec.methodKey()}
					if prev, ok := byKey[key]; ok {
						dec = prev
					} else {
						byKey[key] = dec
					}
					dec.targets = append(dec.targets, bc.name)
				}
			}
		}
	}

	decls := make([]*decl, 0, len(byKey))
	for _, d := range byKey {
		slices.Sort(d.targets)
		d.targets = slices.Compact(d.targets)
		decls = append(decls, d)
	}
	slices.SortFunc(decls, func(a, b *decl) int {
		return cmp.Or(cmp.Compare(a.file, b.file), cmp.Compare(a.recv, b.recv), cmp.Compare(a.name, b.name))
	})
	return decls
}

func declOf(fd *ast.FuncDecl, pkg, file, gen string) *decl {
	d := &decl{pkg: pkg, file: file, gen: gen, name: fd.Name.Name}
	if fd.Recv != nil && len(fd.Recv.List) == 1 {
		// Receivers in this API are always by value and never generic.
		if id, ok := fd.Recv.List[0].Type.(*ast.Ident); ok {
			d.recv = id.Name
		} else {
			d.recv = types.ExprString(fd.Recv.List[0].Type)
		}
	}
	d.in = fields(fd.Type.Params)
	d.out = fields(fd.Type.Results)
	if fd.Doc != nil {
		var lines []string
		for _, c := range fd.Doc.List {
			lines = append(lines, c.Text)
			if strings.TrimSpace(c.Text) == "//" {
				d.hasSep = true
			}
		}
		d.doc = strings.Join(lines, "\n")
	}
	return d
}

// fields flattens a parameter or result list, preserving declaration order and
// expanding grouped names ("x, y Vec") into one param each.
func fields(fl *ast.FieldList) []param {
	if fl == nil {
		return nil
	}
	var ps []param
	for _, f := range fl.List {
		typ := types.ExprString(f.Type)
		if len(f.Names) == 0 {
			ps = append(ps, param{"", typ})
			continue
		}
		for _, n := range f.Names {
			ps = append(ps, param{n.Name, typ})
		}
	}
	return ps
}

// generatorOf returns the tool named in a "Code generated by" header, or
// "hand-written" if there is none.
//
// Since simdgen runs separately per target, for simdgen this returns
// "simdgen/<arch>", where arch is the -arch flag passed to simdgen.
func generatorOf(f *ast.File) string {
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			if !strings.Contains(c.Text, "Code generated by") {
				continue
			}
			for _, tool := range []string{"simdgen", "tmplgen", "wasmgen", "midway", "refgen"} {
				if !strings.Contains(c.Text, tool) {
					continue
				}
				if args := strings.Fields(c.Text); tool == "simdgen" {
					for i, a := range args {
						if a == "-arch" && i+1 < len(args) {
							return tool + "/" + args[i+1]
						}
					}
				}
				return tool
			}
			return "generated"
		}
		break // only the leading comment block can carry the header
	}
	return "hand-written"
}

// readYAML parses one YAML document into a node tree and returns its root
// content node. The generators' inputs carry custom tags (!sum, !string, and
// the merge keys in comments.yaml), but a node tree keeps a tag as data rather
// than trying to resolve it, so this reads them without knowing their schemas.
//
// A missing file counts as empty: these inputs are deleted by comments-yaml and cat-delete,
// and their figures should read zero rather than crash once they are gone. A
// malformed file is fatal, because that is a real problem with the input.
func readYAML(path string) *yaml.Node {
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	} else if err != nil {
		log.Fatal(err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(b, &doc); err != nil {
		log.Fatalf("%s: %s", path, err)
	}
	if doc.Kind == yaml.DocumentNode && len(doc.Content) == 1 {
		return doc.Content[0]
	}
	return &doc
}

// eachEntry calls fn for every key/value pair of every mapping under n, depth
// first.
//
// It does not follow aliases. comments.yaml merges a shared block into fourteen
// receivers with "<<: *common_methods", and those entries are counted where
// they are defined; expanding the merge would count them fifteen times over.
func eachEntry(n *yaml.Node, fn func(key, val *yaml.Node)) {
	if n == nil || n.Kind == yaml.AliasNode {
		return
	}
	if n.Kind == yaml.MappingNode {
		for i := 0; i+1 < len(n.Content); i += 2 {
			fn(n.Content[i], n.Content[i+1])
		}
	}
	for _, c := range n.Content {
		eachEntry(c, fn)
	}
}

// mapValue returns the value of key in a mapping node, or nil.
func mapValue(n *yaml.Node, key string) *yaml.Node {
	if n == nil || n.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(n.Content); i += 2 {
		if n.Content[i].Value == key {
			return n.Content[i+1]
		}
	}
	return nil
}

// isString reports whether n is a string scalar, which in these files means a
// documentation string. Block scalars ("|-") and quoted scalars both qualify.
func isString(n *yaml.Node) bool {
	return n != nil && n.Kind == yaml.ScalarNode && n.ShortTag() == "!!str"
}

// categoriesInventory holds the inventory of categories.yaml.
type categoriesInventory struct {
	allFields      map[string]int // field → count across all entries
	exportedFields map[string]int // field → count across exported entries only
	entries        int
	uniq           int
	exported       int
	compilerOnly   int
}

func makeCategoriesInventory(genDir string) categoriesInventory {
	r := categoriesInventory{
		allFields:      map[string]int{},
		exportedFields: map[string]int{},
	}
	names := map[string]bool{}
	paths, _ := filepath.Glob(filepath.Join(genDir, "simdgen", "ops", "*", "categories.yaml"))
	slices.Sort(paths)
	for _, p := range paths {
		root := readYAML(p)
		if root == nil || root.Kind != yaml.SequenceNode {
			continue
		}
		for _, entry := range root.Content {
			if entry.Kind != yaml.MappingNode {
				continue
			}
			goName := mapValue(entry, "go")
			if goName == nil {
				continue
			}
			r.entries++
			names[goName.Value] = true
			isExported := goName.Value != "" && !(goName.Value[0] >= 'a' && goName.Value[0] <= 'z')
			hasNoTypes := mapValue(entry, "noTypes") != nil
			if !isExported || hasNoTypes {
				r.compilerOnly++
			}
			eachEntry(entry, func(k, _ *yaml.Node) {
				r.allFields[k.Value]++
				if isExported && !hasNoTypes {
					r.exportedFields[k.Value]++
				}
			})
		}
	}
	for n := range names {
		r.uniq++
		if n != "" && !(n[0] >= 'a' && n[0] <= 'z') {
			r.exported++
		}
	}
	return r
}

// countMidwayComments counts the doc-bearing entries in midway/comments.yaml:
// every key whose value is a string, wherever it appears.
func countMidwayComments(genDir string) (n int) {
	eachEntry(readYAML(commentsPath(genDir)), func(_, v *yaml.Node) {
		if isString(v) {
			n++
		}
	})
	return n
}

// commonMethods returns the method names defined in simd_emulated.go,
// less [nonSpecOp]. That set is spec-common's scope, and the portable
// core a new architecture implements first.
func commonMethods(api []*decl) []string {
	seen := map[string]bool{}
	var names []string
	for _, d := range api {
		if d.file == "simd_emulated.go" && d.recv != "" && !nonSpecOp(d.name) {
			if !seen[d.name] {
				seen[d.name] = true
				names = append(names, d.name)
			}
		}
	}
	slices.Sort(names)
	return names
}

func commentsPath(genDir string) string {
	return filepath.Join(genDir, "midway", "comments.yaml")
}
