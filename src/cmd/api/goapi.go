// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Api computes the exported API of a set of Go packages.
//
// BUG(bradfitz): Note that this tool is only currently suitable
// for use on the Go standard library, not arbitrary packages.
// Once the Go AST has type information, this tool will be more
// reliable without hard-coded hacks throughout.
package main

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/parser"
	"go/printer"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Flags
var (
	checkFile = flag.String("c", "", "optional filename to check API against")
	verbose   = flag.Bool("v", false, "Verbose debugging")
)

var contexts = []*build.Context{
	{GOOS: "linux", GOARCH: "386", CgoEnabled: true},
	{GOOS: "linux", GOARCH: "386"},
	{GOOS: "linux", GOARCH: "amd64", CgoEnabled: true},
	{GOOS: "linux", GOARCH: "amd64"},
	{GOOS: "darwin", GOARCH: "386", CgoEnabled: true},
	{GOOS: "darwin", GOARCH: "386"},
	{GOOS: "darwin", GOARCH: "amd64", CgoEnabled: true},
	{GOOS: "darwin", GOARCH: "amd64"},
	{GOOS: "windows", GOARCH: "amd64"},
	{GOOS: "windows", GOARCH: "386"},
}

func init() {
	for _, c := range contexts {
		c.Compiler = build.Default.Compiler
	}
}

func contextName(c *build.Context) string {
	s := c.GOOS + "-" + c.GOARCH
	if c.CgoEnabled {
		return s + "-cgo"
	}
	return s
}

func main() {
	flag.Parse()

	var pkgs []string
	if flag.NArg() > 0 {
		pkgs = flag.Args()
	} else {
		stds, err := exec.Command("go", "list", "std").Output()
		if err != nil {
			log.Fatal(err)
		}
		pkgs = strings.Fields(string(stds))
	}

	var featureCtx = make(map[string]map[string]bool) // feature -> context name -> true
	for _, context := range contexts {
		w := NewWalker()
		w.context = context

		for _, pkg := range pkgs {
			w.wantedPkg[pkg] = true
		}

		for _, pkg := range pkgs {
			if strings.HasPrefix(pkg, "cmd/") ||
				strings.HasPrefix(pkg, "exp/") ||
				strings.HasPrefix(pkg, "old/") {
				continue
			}
			if fi, err := os.Stat(filepath.Join(w.root, pkg)); err != nil || !fi.IsDir() {
				log.Fatalf("no source in tree for package %q", pkg)
			}
			w.WalkPackage(pkg)
		}
		ctxName := contextName(context)
		for _, f := range w.Features() {
			if featureCtx[f] == nil {
				featureCtx[f] = make(map[string]bool)
			}
			featureCtx[f][ctxName] = true
		}
	}

	var features []string
	for f, cmap := range featureCtx {
		if len(cmap) == len(contexts) {
			features = append(features, f)
			continue
		}
		comma := strings.Index(f, ",")
		for cname := range cmap {
			f2 := fmt.Sprintf("%s (%s)%s", f[:comma], cname, f[comma:])
			features = append(features, f2)
		}
	}
	sort.Strings(features)

	bw := bufio.NewWriter(os.Stdout)
	defer bw.Flush()

	if *checkFile != "" {
		bs, err := ioutil.ReadFile(*checkFile)
		if err != nil {
			log.Fatalf("Error reading file %s: %v", *checkFile, err)
		}
		v1 := strings.Split(string(bs), "\n")
		sort.Strings(v1)
		v2 := features
		take := func(sl *[]string) string {
			s := (*sl)[0]
			*sl = (*sl)[1:]
			return s
		}
		for len(v1) > 0 || len(v2) > 0 {
			switch {
			case len(v2) == 0 || v1[0] < v2[0]:
				fmt.Fprintf(bw, "-%s\n", take(&v1))
			case len(v1) == 0 || v1[0] > v2[0]:
				fmt.Fprintf(bw, "+%s\n", take(&v2))
			default:
				take(&v1)
				take(&v2)
			}
		}
	} else {
		for _, f := range features {
			fmt.Fprintf(bw, "%s\n", f)
		}
	}
}

// pkgSymbol represents a symbol in a package
type pkgSymbol struct {
	pkg    string // "net/http"
	symbol string // "RoundTripper"
}

type Walker struct {
	context         *build.Context
	root            string
	fset            *token.FileSet
	scope           []string
	features        map[string]bool // set
	lastConstType   string
	curPackageName  string
	curPackage      *ast.Package
	prevConstType   map[pkgSymbol]string
	constDep        map[string]string // key's const identifier has type of future value const identifier
	packageState    map[string]loadState
	interfaces      map[pkgSymbol]*ast.InterfaceType
	functionTypes   map[pkgSymbol]string // symbol => return type
	selectorFullPkg map[string]string    // "http" => "net/http", updated by imports
	wantedPkg       map[string]bool      // packages requested on the command line
}

func NewWalker() *Walker {
	return &Walker{
		fset:            token.NewFileSet(),
		features:        make(map[string]bool),
		packageState:    make(map[string]loadState),
		interfaces:      make(map[pkgSymbol]*ast.InterfaceType),
		functionTypes:   make(map[pkgSymbol]string),
		selectorFullPkg: make(map[string]string),
		wantedPkg:       make(map[string]bool),
		prevConstType:   make(map[pkgSymbol]string),
		root:            filepath.Join(build.Default.GOROOT, "src/pkg"),
	}
}

// loadState is the state of a package's parsing.
type loadState int

const (
	notLoaded loadState = iota
	loading
	loaded
)

// hardCodedConstantType is a hack until the type checker is sufficient for our needs.
// Rather than litter the code with unnecessary type annotations, we'll hard-code
// the cases we can't handle yet.
func (w *Walker) hardCodedConstantType(name string) (typ string, ok bool) {
	switch w.scope[0] {
	case "pkg syscall":
		switch name {
		case "darwinAMD64":
			return "bool", true
		}
	}
	return "", false
}

func (w *Walker) Features() (fs []string) {
	for f := range w.features {
		fs = append(fs, f)
	}
	sort.Strings(fs)
	return
}

// fileDeps returns the imports in a file.
func fileDeps(f *ast.File) (pkgs []string) {
	for _, is := range f.Imports {
		fpkg, err := strconv.Unquote(is.Path.Value)
		if err != nil {
			log.Fatalf("error unquoting import string %q: %v", is.Path.Value, err)
		}
		if fpkg != "C" {
			pkgs = append(pkgs, fpkg)
		}
	}
	return
}

// WalkPackage walks all files in package `name'.
// WalkPackage does nothing if the package has already been loaded.
func (w *Walker) WalkPackage(name string) {
	switch w.packageState[name] {
	case loading:
		log.Fatalf("import cycle loading package %q?", name)
	case loaded:
		return
	}
	w.packageState[name] = loading
	defer func() {
		w.packageState[name] = loaded
	}()
	dir := filepath.Join(w.root, filepath.FromSlash(name))

	ctxt := w.context
	if ctxt == nil {
		ctxt = &build.Default
	}
	info, err := ctxt.ImportDir(dir, 0)
	if err != nil {
		if strings.Contains(err.Error(), "no Go source files") {
			return
		}
		log.Fatalf("pkg %q, dir %q: ScanDir: %v", name, dir, err)
	}

	apkg := &ast.Package{
		Files: make(map[string]*ast.File),
	}

	files := append(append([]string{}, info.GoFiles...), info.CgoFiles...)
	for _, file := range files {
		f, err := parser.ParseFile(w.fset, filepath.Join(dir, file), nil, 0)
		if err != nil {
			log.Fatalf("error parsing package %s, file %s: %v", name, file, err)
		}
		apkg.Files[file] = f

		for _, dep := range fileDeps(f) {
			w.WalkPackage(dep)
		}
	}

	log.Printf("package %s", name)
	pop := w.pushScope("pkg " + name)
	defer pop()

	w.curPackageName = name
	w.curPackage = apkg
	w.constDep = map[string]string{}

	for _, afile := range apkg.Files {
		w.recordTypes(afile)
	}

	// Register all function declarations first.
	for _, afile := range apkg.Files {
		for _, di := range afile.Decls {
			if d, ok := di.(*ast.FuncDecl); ok {
				w.peekFuncDecl(d)
			}
		}
	}

	for _, afile := range apkg.Files {
		w.walkFile(afile)
	}

	w.resolveConstantDeps()

	// Now that we're done walking types, vars and consts
	// in the *ast.Package, use go/doc to do the rest
	// (functions and methods). This is done here because
	// go/doc is destructive.  We can't use the
	// *ast.Package after this.
	dpkg := doc.New(apkg, name, doc.AllMethods)

	for _, t := range dpkg.Types {
		// Move funcs up to the top-level, not hiding in the Types.
		dpkg.Funcs = append(dpkg.Funcs, t.Funcs...)

		for _, m := range t.Methods {
			w.walkFuncDecl(m.Decl)
		}
	}

	for _, f := range dpkg.Funcs {
		w.walkFuncDecl(f.Decl)
	}
}

// pushScope enters a new scope (walking a package, type, node, etc)
// and returns a function that will leave the scope (with sanity checking
// for mismatched pushes & pops)
func (w *Walker) pushScope(name string) (popFunc func()) {
	w.scope = append(w.scope, name)
	return func() {
		if len(w.scope) == 0 {
			log.Fatalf("attempt to leave scope %q with empty scope list", name)
		}
		if w.scope[len(w.scope)-1] != name {
			log.Fatalf("attempt to leave scope %q, but scope is currently %#v", name, w.scope)
		}
		w.scope = w.scope[:len(w.scope)-1]
	}
}

func (w *Walker) recordTypes(file *ast.File) {
	for _, di := range file.Decls {
		switch d := di.(type) {
		case *ast.GenDecl:
			switch d.Tok {
			case token.TYPE:
				for _, sp := range d.Specs {
					ts := sp.(*ast.TypeSpec)
					name := ts.Name.Name
					if ast.IsExported(name) {
						if it, ok := ts.Type.(*ast.InterfaceType); ok {
							w.noteInterface(name, it)
						}
					}
				}
			}
		}
	}
}

func (w *Walker) walkFile(file *ast.File) {
	// Not entering a scope here; file boundaries aren't interesting.
	for _, di := range file.Decls {
		switch d := di.(type) {
		case *ast.GenDecl:
			switch d.Tok {
			case token.IMPORT:
				for _, sp := range d.Specs {
					is := sp.(*ast.ImportSpec)
					fpath, err := strconv.Unquote(is.Path.Value)
					if err != nil {
						log.Fatal(err)
					}
					name := path.Base(fpath)
					if is.Name != nil {
						name = is.Name.Name
					}
					w.selectorFullPkg[name] = fpath
				}
			case token.CONST:
				for _, sp := range d.Specs {
					w.walkConst(sp.(*ast.ValueSpec))
				}
			case token.TYPE:
				for _, sp := range d.Specs {
					w.walkTypeSpec(sp.(*ast.TypeSpec))
				}
			case token.VAR:
				for _, sp := range d.Specs {
					w.walkVar(sp.(*ast.ValueSpec))
				}
			default:
				log.Fatalf("unknown token type %d in GenDecl", d.Tok)
			}
		case *ast.FuncDecl:
			// Ignore. Handled in subsequent pass, by go/doc.
		default:
			log.Printf("unhandled %T, %#v\n", di, di)
			printer.Fprint(os.Stderr, w.fset, di)
			os.Stderr.Write([]byte("\n"))
		}
	}
}

var constType = map[token.Token]string{
	token.INT:    "ideal-int",
	token.FLOAT:  "ideal-float",
	token.STRING: "ideal-string",
	token.CHAR:   "ideal-char",
	token.IMAG:   "ideal-imag",
}

var varType = map[token.Token]string{
	token.INT:    "int",
	token.FLOAT:  "float64",
	token.STRING: "string",
	token.CHAR:   "rune",
	token.IMAG:   "complex128",
}

var errTODO = errors.New("TODO")

func (w *Walker) constValueType(vi interface{}) (string, error) {
	switch v := vi.(type) {
	case *ast.BasicLit:
		litType, ok := constType[v.Kind]
		if !ok {
			return "", fmt.Errorf("unknown basic literal kind %#v", v)
		}
		return litType, nil
	case *ast.UnaryExpr:
		return w.constValueType(v.X)
	case *ast.SelectorExpr:
		lhs := w.nodeString(v.X)
		rhs := w.nodeString(v.Sel)
		pkg, ok := w.selectorFullPkg[lhs]
		if !ok {
			return "", fmt.Errorf("unknown constant reference; unknown package in expression %s.%s", lhs, rhs)
		}
		if t, ok := w.prevConstType[pkgSymbol{pkg, rhs}]; ok {
			return t, nil
		}
		return "", fmt.Errorf("unknown constant reference to %s.%s", lhs, rhs)
	case *ast.Ident:
		if v.Name == "iota" {
			return "ideal-int", nil // hack.
		}
		if v.Name == "false" || v.Name == "true" {
			return "bool", nil
		}
		if v.Name == "intSize" && w.curPackageName == "strconv" {
			// Hack.
			return "ideal-int", nil
		}
		if t, ok := w.prevConstType[pkgSymbol{w.curPackageName, v.Name}]; ok {
			return t, nil
		}
		return constDepPrefix + v.Name, nil
	case *ast.BinaryExpr:
		left, err := w.constValueType(v.X)
		if err != nil {
			return "", err
		}
		right, err := w.constValueType(v.Y)
		if err != nil {
			return "", err
		}
		if left != right {
			// TODO(bradfitz): encode the real rules here,
			// rather than this mess.
			if left == "ideal-int" && right == "ideal-float" {
				return "ideal-float", nil // math.Log2E
			}
			if left == "ideal-char" && right == "ideal-int" {
				return "ideal-int", nil // math/big.MaxBase
			}
			if left == "ideal-int" && right == "ideal-char" {
				return "ideal-int", nil // text/scanner.GoWhitespace
			}
			if left == "ideal-int" && right == "Duration" {
				// Hack, for package time.
				return "Duration", nil
			}
			if left == "ideal-int" && !strings.HasPrefix(right, "ideal-") {
				return right, nil
			}
			if right == "ideal-int" && !strings.HasPrefix(left, "ideal-") {
				return left, nil
			}
			if strings.HasPrefix(left, constDepPrefix) && strings.HasPrefix(right, constDepPrefix) {
				// Just pick one.
				// e.g. text/scanner GoTokens const-dependency:ScanIdents, const-dependency:ScanFloats
				return left, nil
			}
			return "", fmt.Errorf("in BinaryExpr, unhandled type mismatch; left=%q, right=%q", left, right)
		}
		return left, nil
	case *ast.CallExpr:
		// Not a call, but a type conversion.
		return w.nodeString(v.Fun), nil
	case *ast.ParenExpr:
		return w.constValueType(v.X)
	}
	return "", fmt.Errorf("unknown const value type %T", vi)
}

func (w *Walker) varValueType(vi interface{}) (string, error) {
	switch v := vi.(type) {
	case *ast.BasicLit:
		litType, ok := varType[v.Kind]
		if !ok {
			return "", fmt.Errorf("unknown basic literal kind %#v", v)
		}
		return litType, nil
	case *ast.CompositeLit:
		return w.nodeString(v.Type), nil
	case *ast.FuncLit:
		return w.nodeString(w.namelessType(v.Type)), nil
	case *ast.UnaryExpr:
		if v.Op == token.AND {
			typ, err := w.varValueType(v.X)
			return "*" + typ, err
		}
		return "", fmt.Errorf("unknown unary expr: %#v", v)
	case *ast.SelectorExpr:
		return "", errTODO
	case *ast.Ident:
		node, _, ok := w.resolveName(v.Name)
		if !ok {
			return "", fmt.Errorf("unresolved identifier: %q", v.Name)
		}
		return w.varValueType(node)
	case *ast.BinaryExpr:
		left, err := w.varValueType(v.X)
		if err != nil {
			return "", err
		}
		right, err := w.varValueType(v.Y)
		if err != nil {
			return "", err
		}
		if left != right {
			return "", fmt.Errorf("in BinaryExpr, unhandled type mismatch; left=%q, right=%q", left, right)
		}
		return left, nil
	case *ast.ParenExpr:
		return w.varValueType(v.X)
	case *ast.CallExpr:
		var funSym pkgSymbol
		if selnode, ok := v.Fun.(*ast.SelectorExpr); ok {
			// assume it is not a method.
			pkg, ok := w.selectorFullPkg[w.nodeString(selnode.X)]
			if !ok {
				return "", fmt.Errorf("not a package: %s", w.nodeString(selnode.X))
			}
			funSym = pkgSymbol{pkg, selnode.Sel.Name}
			if retType, ok := w.functionTypes[funSym]; ok {
				if ast.IsExported(retType) && pkg != w.curPackageName {
					// otherpkg.F returning an exported type from otherpkg.
					return pkg + "." + retType, nil
				} else {
					return retType, nil
				}
			}
		} else {
			funSym = pkgSymbol{w.curPackageName, w.nodeString(v.Fun)}
			if retType, ok := w.functionTypes[funSym]; ok {
				return retType, nil
			}
		}
		// maybe a function call; maybe a conversion.  Need to lookup type.
		// TODO(bradfitz): this is a hack, but arguably most of this tool is,
		// until the Go AST has type information.
		nodeStr := w.nodeString(v.Fun)
		switch nodeStr {
		case "string", "[]byte":
			return nodeStr, nil
		}
		return "", fmt.Errorf("not a known function %q", nodeStr)
	default:
		return "", fmt.Errorf("unknown const value type %T", vi)
	}
	panic("unreachable")
}

// resolveName finds a top-level node named name and returns the node
// v and its type t, if known.
func (w *Walker) resolveName(name string) (v interface{}, t interface{}, ok bool) {
	for _, file := range w.curPackage.Files {
		for _, di := range file.Decls {
			switch d := di.(type) {
			case *ast.GenDecl:
				switch d.Tok {
				case token.VAR:
					for _, sp := range d.Specs {
						vs := sp.(*ast.ValueSpec)
						for i, vname := range vs.Names {
							if vname.Name == name {
								if len(vs.Values) > i {
									return vs.Values[i], vs.Type, true
								}
								return nil, vs.Type, true
							}
						}
					}
				}
			}
		}
	}
	return nil, nil, false
}

// constDepPrefix is a magic prefix that is used by constValueType
// and walkConst to signal that a type isn't known yet. These are
// resolved at the end of walking of a package's files.
const constDepPrefix = "const-dependency:"

func (w *Walker) walkConst(vs *ast.ValueSpec) {
	for _, ident := range vs.Names {
		litType := ""
		if vs.Type != nil {
			litType = w.nodeString(vs.Type)
		} else {
			litType = w.lastConstType
			if vs.Values != nil {
				if len(vs.Values) != 1 {
					log.Fatalf("const %q, values: %#v", ident.Name, vs.Values)
				}
				var err error
				litType, err = w.constValueType(vs.Values[0])
				if err != nil {
					if t, ok := w.hardCodedConstantType(ident.Name); ok {
						litType = t
						err = nil
					} else {
						log.Fatalf("unknown kind in const %q (%T): %v", ident.Name, vs.Values[0], err)
					}
				}
			}
		}
		if strings.HasPrefix(litType, constDepPrefix) {
			dep := litType[len(constDepPrefix):]
			w.constDep[ident.Name] = dep
			continue
		}
		if litType == "" {
			log.Fatalf("unknown kind in const %q", ident.Name)
		}
		w.lastConstType = litType

		w.prevConstType[pkgSymbol{w.curPackageName, ident.Name}] = litType

		if ast.IsExported(ident.Name) {
			w.emitFeature(fmt.Sprintf("const %s %s", ident, litType))
		}
	}
}

func (w *Walker) resolveConstantDeps() {
	var findConstType func(string) string
	findConstType = func(ident string) string {
		if dep, ok := w.constDep[ident]; ok {
			return findConstType(dep)
		}
		if t, ok := w.prevConstType[pkgSymbol{w.curPackageName, ident}]; ok {
			return t
		}
		return ""
	}
	for ident := range w.constDep {
		if !ast.IsExported(ident) {
			continue
		}
		t := findConstType(ident)
		if t == "" {
			log.Fatalf("failed to resolve constant %q", ident)
		}
		w.emitFeature(fmt.Sprintf("const %s %s", ident, t))
	}
}

func (w *Walker) walkVar(vs *ast.ValueSpec) {
	for i, ident := range vs.Names {
		if !ast.IsExported(ident.Name) {
			continue
		}

		typ := ""
		if vs.Type != nil {
			typ = w.nodeString(vs.Type)
		} else {
			if len(vs.Values) == 0 {
				log.Fatalf("no values for var %q", ident.Name)
			}
			if len(vs.Values) > 1 {
				log.Fatalf("more than 1 values in ValueSpec not handled, var %q", ident.Name)
			}
			var err error
			typ, err = w.varValueType(vs.Values[i])
			if err != nil {
				log.Fatalf("unknown type of variable %q, type %T, error = %v\ncode: %s",
					ident.Name, vs.Values[i], err, w.nodeString(vs.Values[i]))
			}
		}
		w.emitFeature(fmt.Sprintf("var %s %s", ident, typ))
	}
}

func (w *Walker) nodeString(node interface{}) string {
	if node == nil {
		return ""
	}
	var b bytes.Buffer
	printer.Fprint(&b, w.fset, node)
	return b.String()
}

func (w *Walker) nodeDebug(node interface{}) string {
	if node == nil {
		return ""
	}
	var b bytes.Buffer
	ast.Fprint(&b, w.fset, node, nil)
	return b.String()
}

func (w *Walker) noteInterface(name string, it *ast.InterfaceType) {
	w.interfaces[pkgSymbol{w.curPackageName, name}] = it
}

func (w *Walker) walkTypeSpec(ts *ast.TypeSpec) {
	name := ts.Name.Name
	if !ast.IsExported(name) {
		return
	}
	switch t := ts.Type.(type) {
	case *ast.StructType:
		w.walkStructType(name, t)
	case *ast.InterfaceType:
		w.walkInterfaceType(name, t)
	default:
		w.emitFeature(fmt.Sprintf("type %s %s", name, w.nodeString(ts.Type)))
	}
}

func (w *Walker) walkStructType(name string, t *ast.StructType) {
	typeStruct := fmt.Sprintf("type %s struct", name)
	w.emitFeature(typeStruct)
	pop := w.pushScope(typeStruct)
	defer pop()
	for _, f := range t.Fields.List {
		typ := f.Type
		for _, name := range f.Names {
			if ast.IsExported(name.Name) {
				w.emitFeature(fmt.Sprintf("%s %s", name, w.nodeString(w.namelessType(typ))))
			}
		}
		if f.Names == nil {
			switch v := typ.(type) {
			case *ast.Ident:
				if ast.IsExported(v.Name) {
					w.emitFeature(fmt.Sprintf("embedded %s", v.Name))
				}
			case *ast.StarExpr:
				switch vv := v.X.(type) {
				case *ast.Ident:
					if ast.IsExported(vv.Name) {
						w.emitFeature(fmt.Sprintf("embedded *%s", vv.Name))
					}
				case *ast.SelectorExpr:
					w.emitFeature(fmt.Sprintf("embedded %s", w.nodeString(typ)))
				default:
					log.Fatalf("unable to handle embedded starexpr before %T", typ)
				}
			case *ast.SelectorExpr:
				w.emitFeature(fmt.Sprintf("embedded %s", w.nodeString(typ)))
			default:
				log.Fatalf("unable to handle embedded %T", typ)
			}
		}
	}
}

// method is a method of an interface.
type method struct {
	name string // "Read"
	sig  string // "([]byte) (int, error)", from funcSigString
}

// interfaceMethods returns the expanded list of methods for an interface.
// pkg is the complete package name ("net/http")
// iname is the interface name.
func (w *Walker) interfaceMethods(pkg, iname string) (methods []method) {
	t, ok := w.interfaces[pkgSymbol{pkg, iname}]
	if !ok {
		log.Fatalf("failed to find interface %s.%s", pkg, iname)
	}

	for _, f := range t.Methods.List {
		typ := f.Type
		switch tv := typ.(type) {
		case *ast.FuncType:
			for _, mname := range f.Names {
				if ast.IsExported(mname.Name) {
					ft := typ.(*ast.FuncType)
					methods = append(methods, method{
						name: mname.Name,
						sig:  w.funcSigString(ft),
					})
				}
			}
		case *ast.Ident:
			embedded := typ.(*ast.Ident).Name
			if embedded == "error" {
				methods = append(methods, method{
					name: "Error",
					sig:  "() string",
				})
				continue
			}
			if !ast.IsExported(embedded) {
				log.Fatalf("unexported embedded interface %q in exported interface %s.%s; confused",
					embedded, pkg, iname)
			}
			methods = append(methods, w.interfaceMethods(pkg, embedded)...)
		case *ast.SelectorExpr:
			lhs := w.nodeString(tv.X)
			rhs := w.nodeString(tv.Sel)
			fpkg, ok := w.selectorFullPkg[lhs]
			if !ok {
				log.Fatalf("can't resolve selector %q in interface %s.%s", lhs, pkg, iname)
			}
			methods = append(methods, w.interfaceMethods(fpkg, rhs)...)
		default:
			log.Fatalf("unknown type %T in interface field", typ)
		}
	}
	return
}

func (w *Walker) walkInterfaceType(name string, t *ast.InterfaceType) {
	methNames := []string{}

	pop := w.pushScope("type " + name + " interface")
	for _, m := range w.interfaceMethods(w.curPackageName, name) {
		methNames = append(methNames, m.name)
		w.emitFeature(fmt.Sprintf("%s%s", m.name, m.sig))
	}
	pop()

	sort.Strings(methNames)
	if len(methNames) == 0 {
		w.emitFeature(fmt.Sprintf("type %s interface {}", name))
	} else {
		w.emitFeature(fmt.Sprintf("type %s interface { %s }", name, strings.Join(methNames, ", ")))
	}
}

func (w *Walker) peekFuncDecl(f *ast.FuncDecl) {
	if f.Recv != nil {
		return
	}
	// Record return type for later use.
	if f.Type.Results != nil && len(f.Type.Results.List) == 1 {
		retType := w.nodeString(w.namelessType(f.Type.Results.List[0].Type))
		w.functionTypes[pkgSymbol{w.curPackageName, f.Name.Name}] = retType
	}
}

func (w *Walker) walkFuncDecl(f *ast.FuncDecl) {
	if !ast.IsExported(f.Name.Name) {
		return
	}
	if f.Recv != nil {
		// Method.
		recvType := w.nodeString(f.Recv.List[0].Type)
		keep := ast.IsExported(recvType) ||
			(strings.HasPrefix(recvType, "*") &&
				ast.IsExported(recvType[1:]))
		if !keep {
			return
		}
		w.emitFeature(fmt.Sprintf("method (%s) %s%s", recvType, f.Name.Name, w.funcSigString(f.Type)))
		return
	}
	// Else, a function
	w.emitFeature(fmt.Sprintf("func %s%s", f.Name.Name, w.funcSigString(f.Type)))
}

func (w *Walker) funcSigString(ft *ast.FuncType) string {
	var b bytes.Buffer
	b.WriteByte('(')
	if ft.Params != nil {
		for i, f := range ft.Params.List {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(w.nodeString(w.namelessType(f.Type)))
		}
	}
	b.WriteByte(')')
	if ft.Results != nil {
		if nr := len(ft.Results.List); nr > 0 {
			b.WriteByte(' ')
			if nr > 1 {
				b.WriteByte('(')
			}
			for i, f := range ft.Results.List {
				if i > 0 {
					b.WriteString(", ")
				}
				b.WriteString(w.nodeString(w.namelessType(f.Type)))
			}
			if nr > 1 {
				b.WriteByte(')')
			}
		}
	}
	return b.String()
}

// namelessType returns a type node that lacks any variable names.
func (w *Walker) namelessType(t interface{}) interface{} {
	ft, ok := t.(*ast.FuncType)
	if !ok {
		return t
	}
	return &ast.FuncType{
		Params:  w.namelessFieldList(ft.Params),
		Results: w.namelessFieldList(ft.Results),
	}
}

// namelessFieldList returns a deep clone of fl, with the cloned fields
// lacking names.
func (w *Walker) namelessFieldList(fl *ast.FieldList) *ast.FieldList {
	fl2 := &ast.FieldList{}
	if fl != nil {
		for _, f := range fl.List {
			fl2.List = append(fl2.List, w.namelessField(f))
		}
	}
	return fl2
}

// namelessField clones f, but not preserving the names of fields.
// (comments and tags are also ignored)
func (w *Walker) namelessField(f *ast.Field) *ast.Field {
	return &ast.Field{
		Type: f.Type,
	}
}

func (w *Walker) emitFeature(feature string) {
	if !w.wantedPkg[w.curPackageName] {
		return
	}
	f := strings.Join(w.scope, ", ") + ", " + feature
	if _, dup := w.features[f]; dup {
		panic("duplicate feature inserted: " + f)
	}

	if strings.Contains(f, "\n") {
		// TODO: for now, just skip over the
		// runtime.MemStatsType.BySize type, which this tool
		// doesn't properly handle. It's pretty low-level,
		// though, so not super important to protect against.
		if strings.HasPrefix(f, "pkg runtime") && strings.Contains(f, "BySize [61]struct") {
			return
		}
		panic("feature contains newlines: " + f)
	}
	w.features[f] = true
	if *verbose {
		log.Printf("feature: %s", f)
	}
}

func strListContains(l []string, s string) bool {
	for _, v := range l {
		if v == s {
			return true
		}
	}
	return false
}
