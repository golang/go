// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package computes the exported API of a set of Go packages.
// It is only a test, not a command, nor a usefully importable package.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"internal/testenv"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

const verbose = false

func goCmd() string {
	var exeSuffix string
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
	path := filepath.Join(testenv.GOROOT(nil), "bin", "go"+exeSuffix)
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return "go"
}

// contexts are the default contexts which are scanned.
var contexts = []*build.Context{
	{GOOS: "linux", GOARCH: "386", CgoEnabled: true},
	{GOOS: "linux", GOARCH: "386"},
	{GOOS: "linux", GOARCH: "amd64", CgoEnabled: true},
	{GOOS: "linux", GOARCH: "amd64"},
	{GOOS: "linux", GOARCH: "arm", CgoEnabled: true},
	{GOOS: "linux", GOARCH: "arm"},
	{GOOS: "darwin", GOARCH: "amd64", CgoEnabled: true},
	{GOOS: "darwin", GOARCH: "amd64"},
	{GOOS: "darwin", GOARCH: "arm64", CgoEnabled: true},
	{GOOS: "darwin", GOARCH: "arm64"},
	{GOOS: "windows", GOARCH: "amd64"},
	{GOOS: "windows", GOARCH: "386"},
	{GOOS: "freebsd", GOARCH: "386", CgoEnabled: true},
	{GOOS: "freebsd", GOARCH: "386"},
	{GOOS: "freebsd", GOARCH: "amd64", CgoEnabled: true},
	{GOOS: "freebsd", GOARCH: "amd64"},
	{GOOS: "freebsd", GOARCH: "arm", CgoEnabled: true},
	{GOOS: "freebsd", GOARCH: "arm"},
	{GOOS: "freebsd", GOARCH: "arm64", CgoEnabled: true},
	{GOOS: "freebsd", GOARCH: "arm64"},
	{GOOS: "freebsd", GOARCH: "riscv64", CgoEnabled: true},
	{GOOS: "freebsd", GOARCH: "riscv64"},
	{GOOS: "netbsd", GOARCH: "386", CgoEnabled: true},
	{GOOS: "netbsd", GOARCH: "386"},
	{GOOS: "netbsd", GOARCH: "amd64", CgoEnabled: true},
	{GOOS: "netbsd", GOARCH: "amd64"},
	{GOOS: "netbsd", GOARCH: "arm", CgoEnabled: true},
	{GOOS: "netbsd", GOARCH: "arm"},
	{GOOS: "netbsd", GOARCH: "arm64", CgoEnabled: true},
	{GOOS: "netbsd", GOARCH: "arm64"},
	{GOOS: "openbsd", GOARCH: "386", CgoEnabled: true},
	{GOOS: "openbsd", GOARCH: "386"},
	{GOOS: "openbsd", GOARCH: "amd64", CgoEnabled: true},
	{GOOS: "openbsd", GOARCH: "amd64"},
}

func contextName(c *build.Context) string {
	s := c.GOOS + "-" + c.GOARCH
	if c.CgoEnabled {
		s += "-cgo"
	}
	if c.Dir != "" {
		s += fmt.Sprintf(" [%s]", c.Dir)
	}
	return s
}

var internalPkg = regexp.MustCompile(`(^|/)internal($|/)`)

var exitCode = 0

func Check(t *testing.T) {
	checkFiles, err := filepath.Glob(filepath.Join(testenv.GOROOT(t), "api/go1*.txt"))
	if err != nil {
		t.Fatal(err)
	}

	var nextFiles []string
	if v := runtime.Version(); strings.Contains(v, "devel") || strings.Contains(v, "beta") {
		next, err := filepath.Glob(filepath.Join(testenv.GOROOT(t), "api/next/*.txt"))
		if err != nil {
			t.Fatal(err)
		}
		nextFiles = next
	}

	for _, c := range contexts {
		c.Compiler = build.Default.Compiler
	}

	walkers := make([]*Walker, len(contexts))
	var wg sync.WaitGroup
	for i, context := range contexts {
		i, context := i, context
		wg.Add(1)
		go func() {
			defer wg.Done()
			walkers[i] = NewWalker(context, filepath.Join(testenv.GOROOT(t), "src"))
		}()
	}
	wg.Wait()

	var featureCtx = make(map[string]map[string]bool) // feature -> context name -> true
	for _, w := range walkers {
		for _, name := range w.stdPackages {
			pkg, err := w.import_(name)
			if _, nogo := err.(*build.NoGoError); nogo {
				continue
			}
			if err != nil {
				log.Fatalf("Import(%q): %v", name, err)
			}
			w.export(pkg)
		}

		ctxName := contextName(w.context)
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

	bw := bufio.NewWriter(os.Stdout)
	defer bw.Flush()

	var required []string
	for _, file := range checkFiles {
		required = append(required, fileFeatures(file, needApproval(file))...)
	}
	for _, file := range nextFiles {
		required = append(required, fileFeatures(file, true)...)
	}
	exception := fileFeatures(filepath.Join(testenv.GOROOT(t), "api/except.txt"), false)

	if exitCode == 1 {
		t.Errorf("API database problems found")
	}
	if !compareAPI(bw, features, required, exception) {
		t.Errorf("API differences found")
	}
}

// export emits the exported package features.
func (w *Walker) export(pkg *apiPackage) {
	if verbose {
		log.Println(pkg)
	}
	pop := w.pushScope("pkg " + pkg.Path())
	w.current = pkg
	w.collectDeprecated()
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		if token.IsExported(name) {
			w.emitObj(scope.Lookup(name))
		}
	}
	pop()
}

func set(items []string) map[string]bool {
	s := make(map[string]bool)
	for _, v := range items {
		s[v] = true
	}
	return s
}

var spaceParensRx = regexp.MustCompile(` \(\S+?\)`)

func featureWithoutContext(f string) string {
	if !strings.Contains(f, "(") {
		return f
	}
	return spaceParensRx.ReplaceAllString(f, "")
}

// portRemoved reports whether the given port-specific API feature is
// okay to no longer exist because its port was removed.
func portRemoved(feature string) bool {
	return strings.Contains(feature, "(darwin-386)") ||
		strings.Contains(feature, "(darwin-386-cgo)")
}

func compareAPI(w io.Writer, features, required, exception []string) (ok bool) {
	ok = true

	featureSet := set(features)
	exceptionSet := set(exception)

	sort.Strings(features)
	sort.Strings(required)

	take := func(sl *[]string) string {
		s := (*sl)[0]
		*sl = (*sl)[1:]
		return s
	}

	for len(features) > 0 || len(required) > 0 {
		switch {
		case len(features) == 0 || (len(required) > 0 && required[0] < features[0]):
			feature := take(&required)
			if exceptionSet[feature] {
				// An "unfortunate" case: the feature was once
				// included in the API (e.g. go1.txt), but was
				// subsequently removed. These are already
				// acknowledged by being in the file
				// "api/except.txt". No need to print them out
				// here.
			} else if portRemoved(feature) {
				// okay.
			} else if featureSet[featureWithoutContext(feature)] {
				// okay.
			} else {
				fmt.Fprintf(w, "-%s\n", feature)
				ok = false // broke compatibility
			}
		case len(required) == 0 || (len(features) > 0 && required[0] > features[0]):
			newFeature := take(&features)
			fmt.Fprintf(w, "+%s\n", newFeature)
			ok = false // feature not in api/next/*
		default:
			take(&required)
			take(&features)
		}
	}

	return ok
}

// aliasReplacer applies type aliases to earlier API files,
// to avoid misleading negative results.
// This makes all the references to os.FileInfo in go1.txt
// be read as if they said fs.FileInfo, since os.FileInfo is now an alias.
// If there are many of these, we could do a more general solution,
// but for now the replacer is fine.
var aliasReplacer = strings.NewReplacer(
	"os.FileInfo", "fs.FileInfo",
	"os.FileMode", "fs.FileMode",
	"os.PathError", "fs.PathError",
)

func fileFeatures(filename string, needApproval bool) []string {
	bs, err := os.ReadFile(filename)
	if err != nil {
		log.Fatal(err)
	}
	s := string(bs)

	// Diagnose common mistakes people make,
	// since there is no apifmt to format these files.
	// The missing final newline is important for the
	// final release step of cat next/*.txt >go1.X.txt.
	// If the files don't end in full lines, the concatenation goes awry.
	if strings.Contains(s, "\r") {
		log.Printf("%s: contains CRLFs", filename)
		exitCode = 1
	}
	if filepath.Base(filename) == "go1.4.txt" {
		// No use for blank lines in api files, except go1.4.txt
		// used them in a reasonable way and we should let it be.
	} else if strings.HasPrefix(s, "\n") || strings.Contains(s, "\n\n") {
		log.Printf("%s: contains a blank line", filename)
		exitCode = 1
	}
	if s == "" {
		log.Printf("%s: empty file", filename)
		exitCode = 1
	} else if s[len(s)-1] != '\n' {
		log.Printf("%s: missing final newline", filename)
		exitCode = 1
	}
	s = aliasReplacer.Replace(s)
	lines := strings.Split(s, "\n")
	var nonblank []string
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if needApproval {
			feature, approval, ok := strings.Cut(line, "#")
			if !ok {
				log.Printf("%s:%d: missing proposal approval\n", filename, i+1)
				exitCode = 1
			} else {
				_, err := strconv.Atoi(approval)
				if err != nil {
					log.Printf("%s:%d: malformed proposal approval #%s\n", filename, i+1, approval)
					exitCode = 1
				}
			}
			line = strings.TrimSpace(feature)
		} else {
			if strings.Contains(line, " #") {
				log.Printf("%s:%d: unexpected approval\n", filename, i+1)
				exitCode = 1
			}
		}
		nonblank = append(nonblank, line)
	}
	return nonblank
}

var fset = token.NewFileSet()

type Walker struct {
	context     *build.Context
	root        string
	scope       []string
	current     *apiPackage
	deprecated  map[token.Pos]bool
	features    map[string]bool              // set
	imported    map[string]*apiPackage       // packages already imported
	stdPackages []string                     // names, omitting "unsafe", internal, and vendored packages
	importMap   map[string]map[string]string // importer dir -> import path -> canonical path
	importDir   map[string]string            // canonical import path -> dir

}

func NewWalker(context *build.Context, root string) *Walker {
	w := &Walker{
		context:  context,
		root:     root,
		features: map[string]bool{},
		imported: map[string]*apiPackage{"unsafe": &apiPackage{Package: types.Unsafe}},
	}
	w.loadImports()
	return w
}

func (w *Walker) Features() (fs []string) {
	for f := range w.features {
		fs = append(fs, f)
	}
	sort.Strings(fs)
	return
}

var parsedFileCache = make(map[string]*ast.File)

func (w *Walker) parseFile(dir, file string) (*ast.File, error) {
	filename := filepath.Join(dir, file)
	if f := parsedFileCache[filename]; f != nil {
		return f, nil
	}

	f, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	parsedFileCache[filename] = f

	return f, nil
}

// Disable before debugging non-obvious errors from the type-checker.
const usePkgCache = true

var (
	pkgCache = map[string]*apiPackage{} // map tagKey to package
	pkgTags  = map[string][]string{}    // map import dir to list of relevant tags
)

// tagKey returns the tag-based key to use in the pkgCache.
// It is a comma-separated string; the first part is dir, the rest tags.
// The satisfied tags are derived from context but only those that
// matter (the ones listed in the tags argument plus GOOS and GOARCH) are used.
// The tags list, which came from go/build's Package.AllTags,
// is known to be sorted.
func tagKey(dir string, context *build.Context, tags []string) string {
	ctags := map[string]bool{
		context.GOOS:   true,
		context.GOARCH: true,
	}
	if context.CgoEnabled {
		ctags["cgo"] = true
	}
	for _, tag := range context.BuildTags {
		ctags[tag] = true
	}
	// TODO: ReleaseTags (need to load default)
	key := dir

	// explicit on GOOS and GOARCH as global cache will use "all" cached packages for
	// an indirect imported package. See https://github.com/golang/go/issues/21181
	// for more detail.
	tags = append(tags, context.GOOS, context.GOARCH)
	sort.Strings(tags)

	for _, tag := range tags {
		if ctags[tag] {
			key += "," + tag
			ctags[tag] = false
		}
	}
	return key
}

type listImports struct {
	stdPackages []string                     // names, omitting "unsafe", internal, and vendored packages
	importDir   map[string]string            // canonical import path → directory
	importMap   map[string]map[string]string // import path → canonical import path
}

var listCache sync.Map // map[string]listImports, keyed by contextName

// listSem is a semaphore restricting concurrent invocations of 'go list'. 'go
// list' has its own internal concurrency, so we use a hard-coded constant (to
// allow the I/O-intensive phases of 'go list' to overlap) instead of scaling
// all the way up to GOMAXPROCS.
var listSem = make(chan semToken, 2)

type semToken struct{}

// loadImports populates w with information about the packages in the standard
// library and the packages they themselves import in w's build context.
//
// The source import path and expanded import path are identical except for vendored packages.
// For example, on return:
//
//	w.importMap["math"] = "math"
//	w.importDir["math"] = "<goroot>/src/math"
//
//	w.importMap["golang.org/x/net/route"] = "vendor/golang.org/x/net/route"
//	w.importDir["vendor/golang.org/x/net/route"] = "<goroot>/src/vendor/golang.org/x/net/route"
//
// Since the set of packages that exist depends on context, the result of
// loadImports also depends on context. However, to improve test running time
// the configuration for each environment is cached across runs.
func (w *Walker) loadImports() {
	if w.context == nil {
		return // test-only Walker; does not use the import map
	}

	name := contextName(w.context)

	imports, ok := listCache.Load(name)
	if !ok {
		listSem <- semToken{}
		defer func() { <-listSem }()

		cmd := exec.Command(goCmd(), "list", "-e", "-deps", "-json", "std")
		cmd.Env = listEnv(w.context)
		if w.context.Dir != "" {
			cmd.Dir = w.context.Dir
		}
		cmd.Stderr = os.Stderr
		out, err := cmd.Output()
		if err != nil {
			log.Fatalf("loading imports: %v\n%s", err, out)
		}

		var stdPackages []string
		importMap := make(map[string]map[string]string)
		importDir := make(map[string]string)
		dec := json.NewDecoder(bytes.NewReader(out))
		for {
			var pkg struct {
				ImportPath, Dir string
				ImportMap       map[string]string
				Standard        bool
			}
			err := dec.Decode(&pkg)
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatalf("go list: invalid output: %v", err)
			}

			// - Package "unsafe" contains special signatures requiring
			//   extra care when printing them - ignore since it is not
			//   going to change w/o a language change.
			// - Internal and vendored packages do not contribute to our
			//   API surface. (If we are running within the "std" module,
			//   vendored dependencies appear as themselves instead of
			//   their "vendor/" standard-library copies.)
			// - 'go list std' does not include commands, which cannot be
			//   imported anyway.
			if ip := pkg.ImportPath; pkg.Standard && ip != "unsafe" && !strings.HasPrefix(ip, "vendor/") && !internalPkg.MatchString(ip) {
				stdPackages = append(stdPackages, ip)
			}
			importDir[pkg.ImportPath] = pkg.Dir
			if len(pkg.ImportMap) > 0 {
				importMap[pkg.Dir] = make(map[string]string, len(pkg.ImportMap))
			}
			for k, v := range pkg.ImportMap {
				importMap[pkg.Dir][k] = v
			}
		}

		sort.Strings(stdPackages)
		imports = listImports{
			stdPackages: stdPackages,
			importMap:   importMap,
			importDir:   importDir,
		}
		imports, _ = listCache.LoadOrStore(name, imports)
	}

	li := imports.(listImports)
	w.stdPackages = li.stdPackages
	w.importDir = li.importDir
	w.importMap = li.importMap
}

// listEnv returns the process environment to use when invoking 'go list' for
// the given context.
func listEnv(c *build.Context) []string {
	if c == nil {
		return os.Environ()
	}

	environ := append(os.Environ(),
		"GOOS="+c.GOOS,
		"GOARCH="+c.GOARCH)
	if c.CgoEnabled {
		environ = append(environ, "CGO_ENABLED=1")
	} else {
		environ = append(environ, "CGO_ENABLED=0")
	}
	return environ
}

type apiPackage struct {
	*types.Package
	Files []*ast.File
}

// Importing is a sentinel taking the place in Walker.imported
// for a package that is in the process of being imported.
var importing apiPackage

// Import implements types.Importer.
func (w *Walker) Import(name string) (*types.Package, error) {
	return w.ImportFrom(name, "", 0)
}

// ImportFrom implements types.ImporterFrom.
func (w *Walker) ImportFrom(fromPath, fromDir string, mode types.ImportMode) (*types.Package, error) {
	pkg, err := w.importFrom(fromPath, fromDir, mode)
	if err != nil {
		return nil, err
	}
	return pkg.Package, nil
}

func (w *Walker) import_(name string) (*apiPackage, error) {
	return w.importFrom(name, "", 0)
}

func (w *Walker) importFrom(fromPath, fromDir string, mode types.ImportMode) (*apiPackage, error) {
	name := fromPath
	if canonical, ok := w.importMap[fromDir][fromPath]; ok {
		name = canonical
	}

	pkg := w.imported[name]
	if pkg != nil {
		if pkg == &importing {
			log.Fatalf("cycle importing package %q", name)
		}
		return pkg, nil
	}
	w.imported[name] = &importing

	// Determine package files.
	dir := w.importDir[name]
	if dir == "" {
		dir = filepath.Join(w.root, filepath.FromSlash(name))
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		log.Panicf("no source in tree for import %q (from import %s in %s): %v", name, fromPath, fromDir, err)
	}

	context := w.context
	if context == nil {
		context = &build.Default
	}

	// Look in cache.
	// If we've already done an import with the same set
	// of relevant tags, reuse the result.
	var key string
	if usePkgCache {
		if tags, ok := pkgTags[dir]; ok {
			key = tagKey(dir, context, tags)
			if pkg := pkgCache[key]; pkg != nil {
				w.imported[name] = pkg
				return pkg, nil
			}
		}
	}

	info, err := context.ImportDir(dir, 0)
	if err != nil {
		if _, nogo := err.(*build.NoGoError); nogo {
			return nil, err
		}
		log.Fatalf("pkg %q, dir %q: ScanDir: %v", name, dir, err)
	}

	// Save tags list first time we see a directory.
	if usePkgCache {
		if _, ok := pkgTags[dir]; !ok {
			pkgTags[dir] = info.AllTags
			key = tagKey(dir, context, info.AllTags)
		}
	}

	filenames := append(append([]string{}, info.GoFiles...), info.CgoFiles...)

	// Parse package files.
	var files []*ast.File
	for _, file := range filenames {
		f, err := w.parseFile(dir, file)
		if err != nil {
			log.Fatalf("error parsing package %s: %s", name, err)
		}
		files = append(files, f)
	}

	// Type-check package files.
	var sizes types.Sizes
	if w.context != nil {
		sizes = types.SizesFor(w.context.Compiler, w.context.GOARCH)
	}
	conf := types.Config{
		IgnoreFuncBodies: true,
		FakeImportC:      true,
		Importer:         w,
		Sizes:            sizes,
	}
	tpkg, err := conf.Check(name, fset, files, nil)
	if err != nil {
		ctxt := "<no context>"
		if w.context != nil {
			ctxt = fmt.Sprintf("%s-%s", w.context.GOOS, w.context.GOARCH)
		}
		log.Fatalf("error typechecking package %s: %s (%s)", name, err, ctxt)
	}
	pkg = &apiPackage{tpkg, files}

	if usePkgCache {
		pkgCache[key] = pkg
	}

	w.imported[name] = pkg
	return pkg, nil
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

func sortedMethodNames(typ *types.Interface) []string {
	n := typ.NumMethods()
	list := make([]string, n)
	for i := range list {
		list[i] = typ.Method(i).Name()
	}
	sort.Strings(list)
	return list
}

// sortedEmbeddeds returns constraint types embedded in an
// interface. It does not include embedded interface types or methods.
func (w *Walker) sortedEmbeddeds(typ *types.Interface) []string {
	n := typ.NumEmbeddeds()
	list := make([]string, 0, n)
	for i := 0; i < n; i++ {
		emb := typ.EmbeddedType(i)
		switch emb := emb.(type) {
		case *types.Interface:
			list = append(list, w.sortedEmbeddeds(emb)...)
		case *types.Union:
			var buf bytes.Buffer
			nu := emb.Len()
			for i := 0; i < nu; i++ {
				if i > 0 {
					buf.WriteString(" | ")
				}
				term := emb.Term(i)
				if term.Tilde() {
					buf.WriteByte('~')
				}
				w.writeType(&buf, term.Type())
			}
			list = append(list, buf.String())
		}
	}
	sort.Strings(list)
	return list
}

func (w *Walker) writeType(buf *bytes.Buffer, typ types.Type) {
	switch typ := typ.(type) {
	case *types.Basic:
		s := typ.Name()
		switch typ.Kind() {
		case types.UnsafePointer:
			s = "unsafe.Pointer"
		case types.UntypedBool:
			s = "ideal-bool"
		case types.UntypedInt:
			s = "ideal-int"
		case types.UntypedRune:
			// "ideal-char" for compatibility with old tool
			// TODO(gri) change to "ideal-rune"
			s = "ideal-char"
		case types.UntypedFloat:
			s = "ideal-float"
		case types.UntypedComplex:
			s = "ideal-complex"
		case types.UntypedString:
			s = "ideal-string"
		case types.UntypedNil:
			panic("should never see untyped nil type")
		default:
			switch s {
			case "byte":
				s = "uint8"
			case "rune":
				s = "int32"
			}
		}
		buf.WriteString(s)

	case *types.Array:
		fmt.Fprintf(buf, "[%d]", typ.Len())
		w.writeType(buf, typ.Elem())

	case *types.Slice:
		buf.WriteString("[]")
		w.writeType(buf, typ.Elem())

	case *types.Struct:
		buf.WriteString("struct")

	case *types.Pointer:
		buf.WriteByte('*')
		w.writeType(buf, typ.Elem())

	case *types.Tuple:
		panic("should never see a tuple type")

	case *types.Signature:
		buf.WriteString("func")
		w.writeSignature(buf, typ)

	case *types.Interface:
		buf.WriteString("interface{")
		if typ.NumMethods() > 0 || typ.NumEmbeddeds() > 0 {
			buf.WriteByte(' ')
		}
		if typ.NumMethods() > 0 {
			buf.WriteString(strings.Join(sortedMethodNames(typ), ", "))
		}
		if typ.NumEmbeddeds() > 0 {
			buf.WriteString(strings.Join(w.sortedEmbeddeds(typ), ", "))
		}
		if typ.NumMethods() > 0 || typ.NumEmbeddeds() > 0 {
			buf.WriteByte(' ')
		}
		buf.WriteString("}")

	case *types.Map:
		buf.WriteString("map[")
		w.writeType(buf, typ.Key())
		buf.WriteByte(']')
		w.writeType(buf, typ.Elem())

	case *types.Chan:
		var s string
		switch typ.Dir() {
		case types.SendOnly:
			s = "chan<- "
		case types.RecvOnly:
			s = "<-chan "
		case types.SendRecv:
			s = "chan "
		default:
			panic("unreachable")
		}
		buf.WriteString(s)
		w.writeType(buf, typ.Elem())

	case *types.Named:
		obj := typ.Obj()
		pkg := obj.Pkg()
		if pkg != nil && pkg != w.current.Package {
			buf.WriteString(pkg.Name())
			buf.WriteByte('.')
		}
		buf.WriteString(typ.Obj().Name())

	case *types.TypeParam:
		// Type parameter names may change, so use a placeholder instead.
		fmt.Fprintf(buf, "$%d", typ.Index())

	default:
		panic(fmt.Sprintf("unknown type %T", typ))
	}
}

func (w *Walker) writeSignature(buf *bytes.Buffer, sig *types.Signature) {
	if tparams := sig.TypeParams(); tparams != nil {
		w.writeTypeParams(buf, tparams, true)
	}
	w.writeParams(buf, sig.Params(), sig.Variadic())
	switch res := sig.Results(); res.Len() {
	case 0:
		// nothing to do
	case 1:
		buf.WriteByte(' ')
		w.writeType(buf, res.At(0).Type())
	default:
		buf.WriteByte(' ')
		w.writeParams(buf, res, false)
	}
}

func (w *Walker) writeTypeParams(buf *bytes.Buffer, tparams *types.TypeParamList, withConstraints bool) {
	buf.WriteByte('[')
	c := tparams.Len()
	for i := 0; i < c; i++ {
		if i > 0 {
			buf.WriteString(", ")
		}
		tp := tparams.At(i)
		w.writeType(buf, tp)
		if withConstraints {
			buf.WriteByte(' ')
			w.writeType(buf, tp.Constraint())
		}
	}
	buf.WriteByte(']')
}

func (w *Walker) writeParams(buf *bytes.Buffer, t *types.Tuple, variadic bool) {
	buf.WriteByte('(')
	for i, n := 0, t.Len(); i < n; i++ {
		if i > 0 {
			buf.WriteString(", ")
		}
		typ := t.At(i).Type()
		if variadic && i+1 == n {
			buf.WriteString("...")
			typ = typ.(*types.Slice).Elem()
		}
		w.writeType(buf, typ)
	}
	buf.WriteByte(')')
}

func (w *Walker) typeString(typ types.Type) string {
	var buf bytes.Buffer
	w.writeType(&buf, typ)
	return buf.String()
}

func (w *Walker) signatureString(sig *types.Signature) string {
	var buf bytes.Buffer
	w.writeSignature(&buf, sig)
	return buf.String()
}

func (w *Walker) emitObj(obj types.Object) {
	switch obj := obj.(type) {
	case *types.Const:
		if w.isDeprecated(obj) {
			w.emitf("const %s //deprecated", obj.Name())
		}
		w.emitf("const %s %s", obj.Name(), w.typeString(obj.Type()))
		x := obj.Val()
		short := x.String()
		exact := x.ExactString()
		if short == exact {
			w.emitf("const %s = %s", obj.Name(), short)
		} else {
			w.emitf("const %s = %s  // %s", obj.Name(), short, exact)
		}
	case *types.Var:
		if w.isDeprecated(obj) {
			w.emitf("var %s //deprecated", obj.Name())
		}
		w.emitf("var %s %s", obj.Name(), w.typeString(obj.Type()))
	case *types.TypeName:
		w.emitType(obj)
	case *types.Func:
		w.emitFunc(obj)
	default:
		panic("unknown object: " + obj.String())
	}
}

func (w *Walker) emitType(obj *types.TypeName) {
	name := obj.Name()
	if w.isDeprecated(obj) {
		w.emitf("type %s //deprecated", name)
	}
	if tparams := obj.Type().(*types.Named).TypeParams(); tparams != nil {
		var buf bytes.Buffer
		buf.WriteString(name)
		w.writeTypeParams(&buf, tparams, true)
		name = buf.String()
	}
	typ := obj.Type()
	if obj.IsAlias() {
		w.emitf("type %s = %s", name, w.typeString(typ))
		return
	}
	switch typ := typ.Underlying().(type) {
	case *types.Struct:
		w.emitStructType(name, typ)
	case *types.Interface:
		w.emitIfaceType(name, typ)
		return // methods are handled by emitIfaceType
	default:
		w.emitf("type %s %s", name, w.typeString(typ.Underlying()))
	}

	// emit methods with value receiver
	var methodNames map[string]bool
	vset := types.NewMethodSet(typ)
	for i, n := 0, vset.Len(); i < n; i++ {
		m := vset.At(i)
		if m.Obj().Exported() {
			w.emitMethod(m)
			if methodNames == nil {
				methodNames = make(map[string]bool)
			}
			methodNames[m.Obj().Name()] = true
		}
	}

	// emit methods with pointer receiver; exclude
	// methods that we have emitted already
	// (the method set of *T includes the methods of T)
	pset := types.NewMethodSet(types.NewPointer(typ))
	for i, n := 0, pset.Len(); i < n; i++ {
		m := pset.At(i)
		if m.Obj().Exported() && !methodNames[m.Obj().Name()] {
			w.emitMethod(m)
		}
	}
}

func (w *Walker) emitStructType(name string, typ *types.Struct) {
	typeStruct := fmt.Sprintf("type %s struct", name)
	w.emitf(typeStruct)
	defer w.pushScope(typeStruct)()

	for i := 0; i < typ.NumFields(); i++ {
		f := typ.Field(i)
		if !f.Exported() {
			continue
		}
		typ := f.Type()
		if f.Anonymous() {
			if w.isDeprecated(f) {
				w.emitf("embedded %s //deprecated", w.typeString(typ))
			}
			w.emitf("embedded %s", w.typeString(typ))
			continue
		}
		if w.isDeprecated(f) {
			w.emitf("%s //deprecated", f.Name())
		}
		w.emitf("%s %s", f.Name(), w.typeString(typ))
	}
}

func (w *Walker) emitIfaceType(name string, typ *types.Interface) {
	pop := w.pushScope("type " + name + " interface")

	var methodNames []string
	complete := true
	mset := types.NewMethodSet(typ)
	for i, n := 0, mset.Len(); i < n; i++ {
		m := mset.At(i).Obj().(*types.Func)
		if !m.Exported() {
			complete = false
			continue
		}
		methodNames = append(methodNames, m.Name())
		if w.isDeprecated(m) {
			w.emitf("%s //deprecated", m.Name())
		}
		w.emitf("%s%s", m.Name(), w.signatureString(m.Type().(*types.Signature)))
	}

	if !complete {
		// The method set has unexported methods, so all the
		// implementations are provided by the same package,
		// so the method set can be extended. Instead of recording
		// the full set of names (below), record only that there were
		// unexported methods. (If the interface shrinks, we will notice
		// because a method signature emitted during the last loop
		// will disappear.)
		w.emitf("unexported methods")
	}

	pop()

	if !complete {
		return
	}

	if len(methodNames) == 0 {
		w.emitf("type %s interface {}", name)
		return
	}

	sort.Strings(methodNames)
	w.emitf("type %s interface { %s }", name, strings.Join(methodNames, ", "))
}

func (w *Walker) emitFunc(f *types.Func) {
	sig := f.Type().(*types.Signature)
	if sig.Recv() != nil {
		panic("method considered a regular function: " + f.String())
	}
	if w.isDeprecated(f) {
		w.emitf("func %s //deprecated", f.Name())
	}
	w.emitf("func %s%s", f.Name(), w.signatureString(sig))
}

func (w *Walker) emitMethod(m *types.Selection) {
	sig := m.Type().(*types.Signature)
	recv := sig.Recv().Type()
	// report exported methods with unexported receiver base type
	if true {
		base := recv
		if p, _ := recv.(*types.Pointer); p != nil {
			base = p.Elem()
		}
		if obj := base.(*types.Named).Obj(); !obj.Exported() {
			log.Fatalf("exported method with unexported receiver base type: %s", m)
		}
	}
	tps := ""
	if rtp := sig.RecvTypeParams(); rtp != nil {
		var buf bytes.Buffer
		w.writeTypeParams(&buf, rtp, false)
		tps = buf.String()
	}
	if w.isDeprecated(m.Obj()) {
		w.emitf("method (%s%s) %s //deprecated", w.typeString(recv), tps, m.Obj().Name())
	}
	w.emitf("method (%s%s) %s%s", w.typeString(recv), tps, m.Obj().Name(), w.signatureString(sig))
}

func (w *Walker) emitf(format string, args ...any) {
	f := strings.Join(w.scope, ", ") + ", " + fmt.Sprintf(format, args...)
	if strings.Contains(f, "\n") {
		panic("feature contains newlines: " + f)
	}

	if _, dup := w.features[f]; dup {
		panic("duplicate feature inserted: " + f)
	}
	w.features[f] = true

	if verbose {
		log.Printf("feature: %s", f)
	}
}

func needApproval(filename string) bool {
	name := filepath.Base(filename)
	if name == "go1.txt" {
		return false
	}
	minor := strings.TrimSuffix(strings.TrimPrefix(name, "go1."), ".txt")
	n, err := strconv.Atoi(minor)
	if err != nil {
		log.Fatalf("unexpected api file: %v", name)
	}
	return n >= 19 // started tracking approvals in Go 1.19
}

func (w *Walker) collectDeprecated() {
	isDeprecated := func(doc *ast.CommentGroup) bool {
		if doc != nil {
			for _, c := range doc.List {
				if strings.HasPrefix(c.Text, "// Deprecated:") {
					return true
				}
			}
		}
		return false
	}

	w.deprecated = make(map[token.Pos]bool)
	mark := func(id *ast.Ident) {
		if id != nil {
			w.deprecated[id.Pos()] = true
		}
	}
	for _, file := range w.current.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.File:
				if isDeprecated(n.Doc) {
					mark(n.Name)
				}
				return true
			case *ast.GenDecl:
				if isDeprecated(n.Doc) {
					for _, spec := range n.Specs {
						switch spec := spec.(type) {
						case *ast.ValueSpec:
							for _, id := range spec.Names {
								mark(id)
							}
						case *ast.TypeSpec:
							mark(spec.Name)
						}
					}
				}
				return true // look at specs
			case *ast.FuncDecl:
				if isDeprecated(n.Doc) {
					mark(n.Name)
				}
				return false
			case *ast.TypeSpec:
				if isDeprecated(n.Doc) {
					mark(n.Name)
				}
				return true // recurse into struct or interface type
			case *ast.StructType:
				return true // recurse into fields
			case *ast.InterfaceType:
				return true // recurse into methods
			case *ast.FieldList:
				return true // recurse into fields
			case *ast.ValueSpec:
				if isDeprecated(n.Doc) {
					for _, id := range n.Names {
						mark(id)
					}
				}
				return false
			case *ast.Field:
				if isDeprecated(n.Doc) {
					for _, id := range n.Names {
						mark(id)
					}
					if len(n.Names) == 0 {
						// embedded field T or *T?
						typ := n.Type
						if ptr, ok := typ.(*ast.StarExpr); ok {
							typ = ptr.X
						}
						if id, ok := typ.(*ast.Ident); ok {
							mark(id)
						}
					}
				}
				return false
			default:
				return false
			}
		})
	}
}

func (w *Walker) isDeprecated(obj types.Object) bool {
	return w.deprecated[obj.Pos()]
}
