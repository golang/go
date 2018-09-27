// +build ignore

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Vet is a simple checker for static errors in Go source code.
// See doc.go for more information.

package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"cmd/internal/objabi"
)

// Important! If you add flags here, make sure to update cmd/go/internal/vet/vetflag.go.

var (
	verbose = flag.Bool("v", false, "verbose")
	source  = flag.Bool("source", false, "import from source instead of compiled object files")
	tags    = flag.String("tags", "", "space-separated list of build tags to apply when parsing")
	tagList = []string{} // exploded version of tags flag; set in main

	vcfg          vetConfig
	mustTypecheck bool
)

var exitCode = 0

// "-all" flag enables all non-experimental checks
var all = triStateFlag("all", unset, "enable all non-experimental checks")

// Flags to control which individual checks to perform.
var report = map[string]*triState{
	// Only unusual checks are written here.
	// Most checks that operate during the AST walk are added by register.
	"asmdecl":   triStateFlag("asmdecl", unset, "check assembly against Go declarations"),
	"buildtags": triStateFlag("buildtags", unset, "check that +build tags are valid"),
}

// experimental records the flags enabling experimental features. These must be
// requested explicitly; they are not enabled by -all.
var experimental = map[string]bool{}

// setTrueCount record how many flags are explicitly set to true.
var setTrueCount int

// dirsRun and filesRun indicate whether the vet is applied to directory or
// file targets. The distinction affects which checks are run.
var dirsRun, filesRun bool

// includesNonTest indicates whether the vet is applied to non-test targets.
// Certain checks are relevant only if they touch both test and non-test files.
var includesNonTest bool

// A triState is a boolean that knows whether it has been set to either true or false.
// It is used to identify if a flag appears; the standard boolean flag cannot
// distinguish missing from unset. It also satisfies flag.Value.
type triState int

const (
	unset triState = iota
	setTrue
	setFalse
)

func triStateFlag(name string, value triState, usage string) *triState {
	flag.Var(&value, name, usage)
	return &value
}

// triState implements flag.Value, flag.Getter, and flag.boolFlag.
// They work like boolean flags: we can say vet -printf as well as vet -printf=true
func (ts *triState) Get() interface{} {
	return *ts == setTrue
}

func (ts triState) isTrue() bool {
	return ts == setTrue
}

func (ts *triState) Set(value string) error {
	b, err := strconv.ParseBool(value)
	if err != nil {
		return err
	}
	if b {
		*ts = setTrue
		setTrueCount++
	} else {
		*ts = setFalse
	}
	return nil
}

func (ts *triState) String() string {
	switch *ts {
	case unset:
		return "true" // An unset flag will be set by -all, so defaults to true.
	case setTrue:
		return "true"
	case setFalse:
		return "false"
	}
	panic("not reached")
}

func (ts triState) IsBoolFlag() bool {
	return true
}

// vet tells whether to report errors for the named check, a flag name.
func vet(name string) bool {
	return report[name].isTrue()
}

// setExit sets the value for os.Exit when it is called, later. It
// remembers the highest value.
func setExit(err int) {
	if err > exitCode {
		exitCode = err
	}
}

var (
	// Each of these vars has a corresponding case in (*File).Visit.
	assignStmt    *ast.AssignStmt
	binaryExpr    *ast.BinaryExpr
	callExpr      *ast.CallExpr
	compositeLit  *ast.CompositeLit
	exprStmt      *ast.ExprStmt
	forStmt       *ast.ForStmt
	funcDecl      *ast.FuncDecl
	funcLit       *ast.FuncLit
	genDecl       *ast.GenDecl
	interfaceType *ast.InterfaceType
	rangeStmt     *ast.RangeStmt
	returnStmt    *ast.ReturnStmt
	structType    *ast.StructType

	// checkers is a two-level map.
	// The outer level is keyed by a nil pointer, one of the AST vars above.
	// The inner level is keyed by checker name.
	checkers    = make(map[ast.Node]map[string]func(*File, ast.Node))
	pkgCheckers = make(map[string]func(*Package))
	exporters   = make(map[string]func() interface{})
)

// The exporters data as written to the vetx output file.
type vetxExport struct {
	Name string
	Data interface{}
}

// Vet can provide its own "export information"
// about package A to future invocations of vet
// on packages importing A. If B imports A,
// then running "go vet B" actually invokes vet twice:
// first, it runs vet on A, in "vetx-only" mode, which
// skips most checks and only computes export data
// describing A. Then it runs vet on B, making A's vetx
// data available for consultation. The vet of B
// computes vetx data for B in addition to its
// usual vet checks.

// register registers the named check function,
// to be called with AST nodes of the given types.
// The registered functions are not called in vetx-only mode.
func register(name, usage string, fn func(*File, ast.Node), types ...ast.Node) {
	report[name] = triStateFlag(name, unset, usage)
	for _, typ := range types {
		m := checkers[typ]
		if m == nil {
			m = make(map[string]func(*File, ast.Node))
			checkers[typ] = m
		}
		m[name] = fn
	}
}

// registerPkgCheck registers a package-level checking function,
// to be invoked with the whole package being vetted
// before any of the per-node handlers.
// The registered function fn is called even in vetx-only mode
// (see comment above), so fn must take care not to report
// errors when vcfg.VetxOnly is true.
func registerPkgCheck(name string, fn func(*Package)) {
	pkgCheckers[name] = fn
}

// registerExport registers a function to return vetx export data
// that should be saved and provided to future invocations of vet
// when checking packages importing this one.
// The value returned by fn should be nil or else valid to encode using gob.
// Typically a registerExport call is paired with a call to gob.Register.
func registerExport(name string, fn func() interface{}) {
	exporters[name] = fn
}

// Usage is a replacement usage function for the flags package.
func Usage() {
	fmt.Fprintf(os.Stderr, "Usage of vet:\n")
	fmt.Fprintf(os.Stderr, "\tvet [flags] directory...\n")
	fmt.Fprintf(os.Stderr, "\tvet [flags] files... # Must be a single package\n")
	fmt.Fprintf(os.Stderr, "By default, -all is set and all non-experimental checks are run.\n")
	fmt.Fprintf(os.Stderr, "For more information run\n")
	fmt.Fprintf(os.Stderr, "\tgo doc cmd/vet\n\n")
	fmt.Fprintf(os.Stderr, "Flags:\n")
	flag.PrintDefaults()
	os.Exit(2)
}

// File is a wrapper for the state of a file used in the parser.
// The parse tree walkers are all methods of this type.
type File struct {
	pkg     *Package
	fset    *token.FileSet
	name    string
	content []byte
	file    *ast.File
	b       bytes.Buffer // for use by methods

	// Parsed package "foo" when checking package "foo_test"
	basePkg *Package

	// The keys are the objects that are receivers of a "String()
	// string" method. The value reports whether the method has a
	// pointer receiver.
	// This is used by the recursiveStringer method in print.go.
	stringerPtrs map[*ast.Object]bool

	// Registered checkers to run.
	checkers map[ast.Node][]func(*File, ast.Node)

	// Unreachable nodes; can be ignored in shift check.
	dead map[ast.Node]bool
}

func main() {
	objabi.AddVersionFlag()
	flag.Usage = Usage
	flag.Parse()

	// If any flag is set, we run only those checks requested.
	// If all flag is set true or if no flags are set true, set all the non-experimental ones
	// not explicitly set (in effect, set the "-all" flag).
	if setTrueCount == 0 || *all == setTrue {
		for name, setting := range report {
			if *setting == unset && !experimental[name] {
				*setting = setTrue
			}
		}
	}

	// Accept space-separated tags because that matches
	// the go command's other subcommands.
	// Accept commas because go tool vet traditionally has.
	tagList = strings.Fields(strings.ReplaceAll(*tags, ",", " "))

	initPrintFlags()
	initUnusedFlags()

	if flag.NArg() == 0 {
		Usage()
	}

	// Special case for "go vet" passing an explicit configuration:
	// single argument ending in vet.cfg.
	// Once we have a more general mechanism for obtaining this
	// information from build tools like the go command,
	// vet should be changed to use it. This vet.cfg hack is an
	// experiment to learn about what form that information should take.
	if flag.NArg() == 1 && strings.HasSuffix(flag.Arg(0), "vet.cfg") {
		doPackageCfg(flag.Arg(0))
		os.Exit(exitCode)
	}

	for _, name := range flag.Args() {
		// Is it a directory?
		fi, err := os.Stat(name)
		if err != nil {
			warnf("error walking tree: %s", err)
			continue
		}
		if fi.IsDir() {
			dirsRun = true
		} else {
			filesRun = true
			if !strings.HasSuffix(name, "_test.go") {
				includesNonTest = true
			}
		}
	}
	if dirsRun && filesRun {
		Usage()
	}
	if dirsRun {
		for _, name := range flag.Args() {
			walkDir(name)
		}
		os.Exit(exitCode)
	}
	if doPackage(flag.Args(), nil) == nil {
		warnf("no files checked")
	}
	os.Exit(exitCode)
}

// prefixDirectory places the directory name on the beginning of each name in the list.
func prefixDirectory(directory string, names []string) {
	if directory != "." {
		for i, name := range names {
			names[i] = filepath.Join(directory, name)
		}
	}
}

// vetConfig is the JSON config struct prepared by the Go command.
type vetConfig struct {
	Compiler    string
	Dir         string
	ImportPath  string
	GoFiles     []string
	ImportMap   map[string]string
	PackageFile map[string]string
	Standard    map[string]bool
	PackageVetx map[string]string // map from import path to vetx data file
	VetxOnly    bool              // only compute vetx output; don't run ordinary checks
	VetxOutput  string            // file where vetx output should be written

	SucceedOnTypecheckFailure bool

	imp types.Importer
}

func (v *vetConfig) Import(path string) (*types.Package, error) {
	if v.imp == nil {
		v.imp = importer.For(v.Compiler, v.openPackageFile)
	}
	if path == "unsafe" {
		return v.imp.Import("unsafe")
	}
	p := v.ImportMap[path]
	if p == "" {
		return nil, fmt.Errorf("unknown import path %q", path)
	}
	if v.PackageFile[p] == "" {
		if v.Compiler == "gccgo" && v.Standard[path] {
			// gccgo doesn't have sources for standard library packages,
			// but the importer will do the right thing.
			return v.imp.Import(path)
		}
		return nil, fmt.Errorf("unknown package file for import %q", path)
	}
	return v.imp.Import(p)
}

func (v *vetConfig) openPackageFile(path string) (io.ReadCloser, error) {
	file := v.PackageFile[path]
	if file == "" {
		if v.Compiler == "gccgo" && v.Standard[path] {
			// The importer knows how to handle this.
			return nil, nil
		}
		// Note that path here has been translated via v.ImportMap,
		// unlike in the error in Import above. We prefer the error in
		// Import, but it's worth diagnosing this one too, just in case.
		return nil, fmt.Errorf("unknown package file for %q", path)
	}
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	return f, nil
}

// doPackageCfg analyzes a single package described in a config file.
func doPackageCfg(cfgFile string) {
	js, err := ioutil.ReadFile(cfgFile)
	if err != nil {
		errorf("%v", err)
	}
	if err := json.Unmarshal(js, &vcfg); err != nil {
		errorf("parsing vet config %s: %v", cfgFile, err)
	}
	stdImporter = &vcfg
	inittypes()
	mustTypecheck = true
	doPackage(vcfg.GoFiles, nil)
	if vcfg.VetxOutput != "" {
		out := make([]vetxExport, 0, len(exporters))
		for name, fn := range exporters {
			out = append(out, vetxExport{
				Name: name,
				Data: fn(),
			})
		}
		// Sort the data so that it is consistent across builds.
		sort.Slice(out, func(i, j int) bool {
			return out[i].Name < out[j].Name
		})
		var buf bytes.Buffer
		if err := gob.NewEncoder(&buf).Encode(out); err != nil {
			errorf("encoding vet output: %v", err)
			return
		}
		if err := ioutil.WriteFile(vcfg.VetxOutput, buf.Bytes(), 0666); err != nil {
			errorf("saving vet output: %v", err)
			return
		}
	}
}

// doPackageDir analyzes the single package found in the directory, if there is one,
// plus a test package, if there is one.
func doPackageDir(directory string) {
	context := build.Default
	if len(context.BuildTags) != 0 {
		warnf("build tags %s previously set", context.BuildTags)
	}
	context.BuildTags = append(tagList, context.BuildTags...)

	pkg, err := context.ImportDir(directory, 0)
	if err != nil {
		// If it's just that there are no go source files, that's fine.
		if _, nogo := err.(*build.NoGoError); nogo {
			return
		}
		// Non-fatal: we are doing a recursive walk and there may be other directories.
		warnf("cannot process directory %s: %s", directory, err)
		return
	}
	var names []string
	names = append(names, pkg.GoFiles...)
	names = append(names, pkg.CgoFiles...)
	names = append(names, pkg.TestGoFiles...) // These are also in the "foo" package.
	names = append(names, pkg.SFiles...)
	prefixDirectory(directory, names)
	basePkg := doPackage(names, nil)
	// Is there also a "foo_test" package? If so, do that one as well.
	if len(pkg.XTestGoFiles) > 0 {
		names = pkg.XTestGoFiles
		prefixDirectory(directory, names)
		doPackage(names, basePkg)
	}
}

type Package struct {
	path      string
	defs      map[*ast.Ident]types.Object
	uses      map[*ast.Ident]types.Object
	implicits map[ast.Node]types.Object
	selectors map[*ast.SelectorExpr]*types.Selection
	types     map[ast.Expr]types.TypeAndValue
	spans     map[types.Object]Span
	files     []*File
	typesPkg  *types.Package
}

// doPackage analyzes the single package constructed from the named files.
// It returns the parsed Package or nil if none of the files have been checked.
func doPackage(names []string, basePkg *Package) *Package {
	var files []*File
	var astFiles []*ast.File
	fs := token.NewFileSet()
	for _, name := range names {
		data, err := ioutil.ReadFile(name)
		if err != nil {
			// Warn but continue to next package.
			warnf("%s: %s", name, err)
			return nil
		}
		var parsedFile *ast.File
		if strings.HasSuffix(name, ".go") {
			parsedFile, err = parser.ParseFile(fs, name, data, parser.ParseComments)
			if err != nil {
				warnf("%s: %s", name, err)
				return nil
			}
			astFiles = append(astFiles, parsedFile)
		}
		file := &File{
			fset:    fs,
			content: data,
			name:    name,
			file:    parsedFile,
			dead:    make(map[ast.Node]bool),
		}
		files = append(files, file)
	}
	if len(astFiles) == 0 {
		return nil
	}
	pkg := new(Package)
	pkg.path = astFiles[0].Name.Name
	pkg.files = files
	// Type check the package.
	errs := pkg.check(fs, astFiles)
	if errs != nil {
		if vcfg.SucceedOnTypecheckFailure {
			os.Exit(0)
		}
		if *verbose || mustTypecheck {
			for _, err := range errs {
				fmt.Fprintf(os.Stderr, "%v\n", err)
			}
			if mustTypecheck {
				// This message could be silenced, and we could just exit,
				// but it might be helpful at least at first to make clear that the
				// above errors are coming from vet and not the compiler
				// (they often look like compiler errors, such as "declared but not used").
				errorf("typecheck failures")
			}
		}
	}

	// Check.
	for _, file := range files {
		file.pkg = pkg
		file.basePkg = basePkg
	}
	for name, fn := range pkgCheckers {
		if vet(name) {
			fn(pkg)
		}
	}
	if vcfg.VetxOnly {
		return pkg
	}

	chk := make(map[ast.Node][]func(*File, ast.Node))
	for typ, set := range checkers {
		for name, fn := range set {
			if vet(name) {
				chk[typ] = append(chk[typ], fn)
			}
		}
	}
	for _, file := range files {
		checkBuildTag(file)
		file.checkers = chk
		if file.file != nil {
			file.walkFile(file.name, file.file)
		}
	}
	return pkg
}

func visit(path string, f os.FileInfo, err error) error {
	if err != nil {
		warnf("walk error: %s", err)
		return err
	}
	// One package per directory. Ignore the files themselves.
	if !f.IsDir() {
		return nil
	}
	doPackageDir(path)
	return nil
}

func (pkg *Package) hasFileWithSuffix(suffix string) bool {
	for _, f := range pkg.files {
		if strings.HasSuffix(f.name, suffix) {
			return true
		}
	}
	return false
}

// walkDir recursively walks the tree looking for Go packages.
func walkDir(root string) {
	filepath.Walk(root, visit)
}

// errorf formats the error to standard error, adding program
// identification and a newline, and exits.
func errorf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "vet: "+format+"\n", args...)
	os.Exit(2)
}

// warnf formats the error to standard error, adding program
// identification and a newline, but does not exit.
func warnf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "vet: "+format+"\n", args...)
	setExit(1)
}

// Println is fmt.Println guarded by -v.
func Println(args ...interface{}) {
	if !*verbose {
		return
	}
	fmt.Println(args...)
}

// Printf is fmt.Printf guarded by -v.
func Printf(format string, args ...interface{}) {
	if !*verbose {
		return
	}
	fmt.Printf(format+"\n", args...)
}

// Bad reports an error and sets the exit code..
func (f *File) Bad(pos token.Pos, args ...interface{}) {
	f.Warn(pos, args...)
	setExit(1)
}

// Badf reports a formatted error and sets the exit code.
func (f *File) Badf(pos token.Pos, format string, args ...interface{}) {
	f.Warnf(pos, format, args...)
	setExit(1)
}

// loc returns a formatted representation of the position.
func (f *File) loc(pos token.Pos) string {
	if pos == token.NoPos {
		return ""
	}
	// Do not print columns. Because the pos often points to the start of an
	// expression instead of the inner part with the actual error, the
	// precision can mislead.
	posn := f.fset.Position(pos)
	return fmt.Sprintf("%s:%d", posn.Filename, posn.Line)
}

// locPrefix returns a formatted representation of the position for use as a line prefix.
func (f *File) locPrefix(pos token.Pos) string {
	if pos == token.NoPos {
		return ""
	}
	return fmt.Sprintf("%s: ", f.loc(pos))
}

// Warn reports an error but does not set the exit code.
func (f *File) Warn(pos token.Pos, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s%s", f.locPrefix(pos), fmt.Sprintln(args...))
}

// Warnf reports a formatted error but does not set the exit code.
func (f *File) Warnf(pos token.Pos, format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s%s\n", f.locPrefix(pos), fmt.Sprintf(format, args...))
}

// walkFile walks the file's tree.
func (f *File) walkFile(name string, file *ast.File) {
	Println("Checking file", name)
	ast.Walk(f, file)
}

// Visit implements the ast.Visitor interface.
func (f *File) Visit(node ast.Node) ast.Visitor {
	f.updateDead(node)
	var key ast.Node
	switch node.(type) {
	case *ast.AssignStmt:
		key = assignStmt
	case *ast.BinaryExpr:
		key = binaryExpr
	case *ast.CallExpr:
		key = callExpr
	case *ast.CompositeLit:
		key = compositeLit
	case *ast.ExprStmt:
		key = exprStmt
	case *ast.ForStmt:
		key = forStmt
	case *ast.FuncDecl:
		key = funcDecl
	case *ast.FuncLit:
		key = funcLit
	case *ast.GenDecl:
		key = genDecl
	case *ast.InterfaceType:
		key = interfaceType
	case *ast.RangeStmt:
		key = rangeStmt
	case *ast.ReturnStmt:
		key = returnStmt
	case *ast.StructType:
		key = structType
	}
	for _, fn := range f.checkers[key] {
		fn(f, node)
	}
	return f
}

// gofmt returns a string representation of the expression.
func (f *File) gofmt(x ast.Expr) string {
	f.b.Reset()
	printer.Fprint(&f.b, f.fset, x)
	return f.b.String()
}

// imported[path][key] is previously written export data.
var imported = make(map[string]map[string]interface{})

// readVetx reads export data written by a previous
// invocation of vet on an imported package (path).
// The key is the name passed to registerExport
// when the data was originally generated.
// readVetx returns nil if the data is unavailable.
func readVetx(path, key string) interface{} {
	if path == "unsafe" || vcfg.ImportPath == "" {
		return nil
	}
	m := imported[path]
	if m == nil {
		file := vcfg.PackageVetx[path]
		if file == "" {
			return nil
		}
		data, err := ioutil.ReadFile(file)
		if err != nil {
			return nil
		}
		var out []vetxExport
		err = gob.NewDecoder(bytes.NewReader(data)).Decode(&out)
		if err != nil {
			return nil
		}
		m = make(map[string]interface{})
		for _, x := range out {
			m[x.Name] = x.Data
		}
		imported[path] = m
	}
	return m[key]
}
