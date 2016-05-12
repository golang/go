// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

// Bundle creates a single-source-file version of a source package
// suitable for inclusion in a particular target package.
//
// Usage:
//
//	bundle [-o file] [-dst path] [-pkg name] [-prefix p] [-import old=new] <src>
//
// The src argument specifies the import path of the package to bundle.
// The bundling of a directory of source files into a single source file
// necessarily imposes a number of constraints.
// The package being bundled must not use cgo; must not use conditional
// file compilation, whether with build tags or system-specific file names
// like code_amd64.go; must not depend on any special comments, which
// may not be preserved; must not use any assembly sources;
// must not use renaming imports; and must not use reflection-based APIs
// that depend on the specific names of types or struct fields.
//
// By default, bundle writes the bundled code to standard output.
// If the -o argument is given, bundle writes to the named file
// and also includes a ``//go:generate'' comment giving the exact
// command line used, for regenerating the file with ``go generate.''
//
// Bundle customizes its output for inclusion in a particular package, the destination package.
// By default bundle assumes the destination is the package in the current directory,
// but the destination package can be specified explicitly using the -dst option,
// which takes an import path as its argument.
// If the source package imports the destination package, bundle will remove
// those imports and rewrite any references to use direct references to the
// corresponding symbols.
// Bundle also must write a package declaration in the output and must
// choose a name to use in that declaration.
// If the -package option is given, bundle uses that name.
// Otherwise, if the -dst option is given, bundle uses the last
// element of the destination import path.
// Otherwise, by default bundle uses the package name found in the
// package sources in the current directory.
//
// To avoid collisions, bundle inserts a prefix at the beginning of
// every package-level const, func, type, and var identifier in src's code,
// updating references accordingly. The default prefix is the package name
// of the source package followed by an underscore. The -prefix option
// specifies an alternate prefix.
//
// Occasionally it is necessary to rewrite imports during the bundling
// process. The -import option, which may be repeated, specifies that
// an import of "old" should be rewritten to import "new" instead.
//
// Example
//
// Bundle archive/zip for inclusion in cmd/dist:
//
//	cd $GOROOT/src/cmd/dist
//	bundle -o zip.go archive/zip
//
// Bundle golang.org/x/net/http2 for inclusion in net/http,
// prefixing all identifiers by "http2" instead of "http2_",
// and rewriting the import "golang.org/x/net/http2/hpack"
// to "internal/golang.org/x/net/http2/hpack":
//
//	cd $GOROOT/src/net/http
//	bundle -o h2_bundle.go \
//		-prefix http2 \
//		-import golang.org/x/net/http2/hpack=internal/golang.org/x/net/http2/hpack \
//		golang.org/x/net/http2
//
// Two ways to update the http2 bundle:
//
//	go generate net/http
//
//	cd $GOROOT/src/net/http
//	go generate
//
// Update both bundles, restricting ``go generate'' to running bundle commands:
//
//	go generate -run bundle cmd/dist net/http
//
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"path"
	"strings"

	"golang.org/x/tools/go/loader"
)

var (
	outputFile = flag.String("o", "", "write output to `file` (default standard output)")
	dstPath    = flag.String("dst", "", "set destination import `path` (default taken from current directory)")
	pkgName    = flag.String("pkg", "", "set destination package `name` (default taken from current directory)")
	prefix     = flag.String("prefix", "", "set bundled identifier prefix to `p` (default source package name + \"_\")")

	importMap = map[string]string{}
)

func init() {
	flag.Var(flagFunc(addImportMap), "import", "rewrite import using `map`, of form old=new (can be repeated)")
}

func addImportMap(s string) {
	if strings.Count(s, "=") != 1 {
		log.Fatal("-import argument must be of the form old=new")
	}
	i := strings.Index(s, "=")
	old, new := s[:i], s[i+1:]
	if old == "" || new == "" {
		log.Fatal("-import argument must be of the form old=new; old and new must be non-empty")
	}
	importMap[old] = new
}

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: bundle [options] <src>\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	log.SetPrefix("bundle: ")
	log.SetFlags(0)

	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) != 1 {
		usage()
	}

	if *dstPath != "" {
		if *pkgName == "" {
			*pkgName = path.Base(*dstPath)
		}
	} else {
		wd, _ := os.Getwd()
		pkg, err := build.ImportDir(wd, 0)
		if err != nil {
			log.Fatalf("cannot find package in current directory: %v", err)
		}
		*dstPath = pkg.ImportPath
		if *pkgName == "" {
			*pkgName = pkg.Name
		}
	}

	code, err := bundle(args[0], *dstPath, *pkgName, *prefix)
	if err != nil {
		log.Fatal(err)
	}
	if *outputFile != "" {
		err := ioutil.WriteFile(*outputFile, code, 0666)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		_, err := os.Stdout.Write(code)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// isStandardImportPath is copied from cmd/go in the standard library.
func isStandardImportPath(path string) bool {
	i := strings.Index(path, "/")
	if i < 0 {
		i = len(path)
	}
	elem := path[:i]
	return !strings.Contains(elem, ".")
}

var ctxt = &build.Default

func bundle(src, dst, dstpkg, prefix string) ([]byte, error) {
	// Load the initial package.
	conf := loader.Config{ParserMode: parser.ParseComments, Build: ctxt}
	conf.TypeCheckFuncBodies = func(p string) bool { return p == src }
	conf.Import(src)

	lprog, err := conf.Load()
	if err != nil {
		return nil, err
	}

	info := lprog.Package(src)
	if prefix == "" {
		pkgName := info.Files[0].Name.Name
		prefix = pkgName + "_"
	}

	objsToUpdate := make(map[types.Object]bool)
	var rename func(from types.Object)
	rename = func(from types.Object) {
		if !objsToUpdate[from] {
			objsToUpdate[from] = true

			// Renaming a type that is used as an embedded field
			// requires renaming the field too. e.g.
			// 	type T int // if we rename this to U..
			// 	var s struct {T}
			// 	print(s.T) // ...this must change too
			if _, ok := from.(*types.TypeName); ok {
				for id, obj := range info.Uses {
					if obj == from {
						if field := info.Defs[id]; field != nil {
							rename(field)
						}
					}
				}
			}
		}
	}

	// Rename each package-level object.
	scope := info.Pkg.Scope()
	for _, name := range scope.Names() {
		rename(scope.Lookup(name))
	}

	var out bytes.Buffer

	fmt.Fprintf(&out, "// Code generated by golang.org/x/tools/cmd/bundle.\n")
	if *outputFile != "" {
		fmt.Fprintf(&out, "//go:generate bundle %s\n", strings.Join(os.Args[1:], " "))
	} else {
		fmt.Fprintf(&out, "//   $ bundle %s\n", strings.Join(os.Args[1:], " "))
	}
	fmt.Fprintf(&out, "\n")

	// Concatenate package comments from all files...
	for _, f := range info.Files {
		if doc := f.Doc.Text(); strings.TrimSpace(doc) != "" {
			for _, line := range strings.Split(doc, "\n") {
				fmt.Fprintf(&out, "// %s\n", line)
			}
		}
	}
	// ...but don't let them become the actual package comment.
	fmt.Fprintln(&out)

	fmt.Fprintf(&out, "package %s\n\n", dstpkg)

	// Print a single declaration that imports all necessary packages.
	// TODO(adonovan):
	// - support renaming imports.
	// - preserve comments from the original import declarations.
	for _, f := range info.Files {
		for _, imp := range f.Imports {
			if imp.Name != nil {
				return nil, fmt.Errorf("%s: renaming imports not supported",
					lprog.Fset.Position(imp.Pos()))
			}
		}
	}

	var pkgStd, pkgExt []string
	for _, p := range info.Pkg.Imports() {
		if p.Path() == dst {
			continue
		}
		x, ok := importMap[p.Path()]
		if !ok {
			x = p.Path()
		}
		if isStandardImportPath(x) {
			pkgStd = append(pkgStd, x)
		} else {
			pkgExt = append(pkgExt, x)
		}
	}

	fmt.Fprintln(&out, "import (")
	for _, p := range pkgStd {
		fmt.Fprintf(&out, "\t%q\n", p)
	}
	if len(pkgExt) > 0 {
		fmt.Fprintln(&out)
		for _, p := range pkgExt {
			fmt.Fprintf(&out, "\t%q\n", p)
		}
	}
	fmt.Fprintln(&out, ")\n")

	// Modify and print each file.
	for _, f := range info.Files {
		// Update renamed identifiers.
		for id, obj := range info.Defs {
			if objsToUpdate[obj] {
				id.Name = prefix + obj.Name()
			}
		}
		for id, obj := range info.Uses {
			if objsToUpdate[obj] {
				id.Name = prefix + obj.Name()
			}
		}

		// For each qualified identifier that refers to the
		// destination package, remove the qualifier.
		// The "@@@." strings are removed in postprocessing.
		ast.Inspect(f, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if id, ok := sel.X.(*ast.Ident); ok {
					if obj, ok := info.Uses[id].(*types.PkgName); ok {
						if obj.Imported().Path() == dst {
							id.Name = "@@@"
						}
					}
				}
			}
			return true
		})

		// Pretty-print package-level declarations.
		// but no package or import declarations.
		//
		// TODO(adonovan): this may cause loss of comments
		// preceding or associated with the package or import
		// declarations or not associated with any declaration.
		// Check.
		var buf bytes.Buffer
		for _, decl := range f.Decls {
			if decl, ok := decl.(*ast.GenDecl); ok && decl.Tok == token.IMPORT {
				continue
			}
			buf.Reset()
			format.Node(&buf, lprog.Fset, decl)
			// Remove each "@@@." in the output.
			// TODO(adonovan): not hygienic.
			out.Write(bytes.Replace(buf.Bytes(), []byte("@@@."), nil, -1))
			out.WriteString("\n\n")
		}
	}

	// Now format the entire thing.
	result, err := format.Source(out.Bytes())
	if err != nil {
		log.Fatalf("formatting failed: %v", err)
	}

	return result, nil
}

type flagFunc func(string)

func (f flagFunc) Set(s string) error {
	f(s)
	return nil
}

func (f flagFunc) String() string { return "" }
