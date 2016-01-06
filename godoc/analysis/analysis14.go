// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

// Package analysis performs type and pointer analysis
// and generates mark-up for the Go source view.
//
// The Run method populates a Result object by running type and
// (optionally) pointer analysis.  The Result object is thread-safe
// and at all times may be accessed by a serving thread, even as it is
// progressively populated as analysis facts are derived.
//
// The Result is a mapping from each godoc file URL
// (e.g. /src/fmt/print.go) to information about that file.  The
// information is a list of HTML markup links and a JSON array of
// structured data values.  Some of the links call client-side
// JavaScript functions that index this array.
//
// The analysis computes mark-up for the following relations:
//
// IMPORTS: for each ast.ImportSpec, the package that it denotes.
//
// RESOLUTION: for each ast.Ident, its kind and type, and the location
// of its definition.
//
// METHOD SETS, IMPLEMENTS: for each ast.Ident defining a named type,
// its method-set, the set of interfaces it implements or is
// implemented by, and its size/align values.
//
// CALLERS, CALLEES: for each function declaration ('func' token), its
// callers, and for each call-site ('(' token), its callees.
//
// CALLGRAPH: the package docs include an interactive viewer for the
// intra-package call graph of "fmt".
//
// CHANNEL PEERS: for each channel operation make/<-/close, the set of
// other channel ops that alias the same channel(s).
//
// ERRORS: for each locus of a frontend (scanner/parser/type) error, the
// location is highlighted in red and hover text provides the compiler
// error message.
//
package analysis // import "golang.org/x/tools/godoc/analysis"

import (
	"fmt"
	"go/build"
	"go/scanner"
	"go/token"
	"html"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/go/exact"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/go/types"
)

// -- links ------------------------------------------------------------

// A Link is an HTML decoration of the bytes [Start, End) of a file.
// Write is called before/after those bytes to emit the mark-up.
type Link interface {
	Start() int
	End() int
	Write(w io.Writer, _ int, start bool) // the godoc.LinkWriter signature
}

// An <a> element.
type aLink struct {
	start, end int    // =godoc.Segment
	title      string // hover text
	onclick    string // JS code (NB: trusted)
	href       string // URL     (NB: trusted)
}

func (a aLink) Start() int { return a.start }
func (a aLink) End() int   { return a.end }
func (a aLink) Write(w io.Writer, _ int, start bool) {
	if start {
		fmt.Fprintf(w, `<a title='%s'`, html.EscapeString(a.title))
		if a.onclick != "" {
			fmt.Fprintf(w, ` onclick='%s'`, html.EscapeString(a.onclick))
		}
		if a.href != "" {
			// TODO(adonovan): I think that in principle, a.href must first be
			// url.QueryEscape'd, but if I do that, a leading slash becomes "%2F",
			// which causes the browser to treat the path as relative, not absolute.
			// WTF?
			fmt.Fprintf(w, ` href='%s'`, html.EscapeString(a.href))
		}
		fmt.Fprintf(w, ">")
	} else {
		fmt.Fprintf(w, "</a>")
	}
}

// An <a class='error'> element.
type errorLink struct {
	start int
	msg   string
}

func (e errorLink) Start() int { return e.start }
func (e errorLink) End() int   { return e.start + 1 }

func (e errorLink) Write(w io.Writer, _ int, start bool) {
	// <span> causes havoc, not sure why, so use <a>.
	if start {
		fmt.Fprintf(w, `<a class='error' title='%s'>`, html.EscapeString(e.msg))
	} else {
		fmt.Fprintf(w, "</a>")
	}
}

// -- fileInfo ---------------------------------------------------------

// FileInfo holds analysis information for the source file view.
// Clients must not mutate it.
type FileInfo struct {
	Data  []interface{} // JSON serializable values
	Links []Link        // HTML link markup
}

// A fileInfo is the server's store of hyperlinks and JSON data for a
// particular file.
type fileInfo struct {
	mu        sync.Mutex
	data      []interface{} // JSON objects
	links     []Link
	sorted    bool
	hasErrors bool // TODO(adonovan): surface this in the UI
}

// addLink adds a link to the Go source file fi.
func (fi *fileInfo) addLink(link Link) {
	fi.mu.Lock()
	fi.links = append(fi.links, link)
	fi.sorted = false
	if _, ok := link.(errorLink); ok {
		fi.hasErrors = true
	}
	fi.mu.Unlock()
}

// addData adds the structured value x to the JSON data for the Go
// source file fi.  Its index is returned.
func (fi *fileInfo) addData(x interface{}) int {
	fi.mu.Lock()
	index := len(fi.data)
	fi.data = append(fi.data, x)
	fi.mu.Unlock()
	return index
}

// get returns the file info in external form.
// Callers must not mutate its fields.
func (fi *fileInfo) get() FileInfo {
	var r FileInfo
	// Copy slices, to avoid races.
	fi.mu.Lock()
	r.Data = append(r.Data, fi.data...)
	if !fi.sorted {
		sort.Sort(linksByStart(fi.links))
		fi.sorted = true
	}
	r.Links = append(r.Links, fi.links...)
	fi.mu.Unlock()
	return r
}

// PackageInfo holds analysis information for the package view.
// Clients must not mutate it.
type PackageInfo struct {
	CallGraph      []*PCGNodeJSON
	CallGraphIndex map[string]int
	Types          []*TypeInfoJSON
}

type pkgInfo struct {
	mu             sync.Mutex
	callGraph      []*PCGNodeJSON
	callGraphIndex map[string]int  // keys are (*ssa.Function).RelString()
	types          []*TypeInfoJSON // type info for exported types
}

func (pi *pkgInfo) setCallGraph(callGraph []*PCGNodeJSON, callGraphIndex map[string]int) {
	pi.mu.Lock()
	pi.callGraph = callGraph
	pi.callGraphIndex = callGraphIndex
	pi.mu.Unlock()
}

func (pi *pkgInfo) addType(t *TypeInfoJSON) {
	pi.mu.Lock()
	pi.types = append(pi.types, t)
	pi.mu.Unlock()
}

// get returns the package info in external form.
// Callers must not mutate its fields.
func (pi *pkgInfo) get() PackageInfo {
	var r PackageInfo
	// Copy slices, to avoid races.
	pi.mu.Lock()
	r.CallGraph = append(r.CallGraph, pi.callGraph...)
	r.CallGraphIndex = pi.callGraphIndex
	r.Types = append(r.Types, pi.types...)
	pi.mu.Unlock()
	return r
}

// -- Result -----------------------------------------------------------

// Result contains the results of analysis.
// The result contains a mapping from filenames to a set of HTML links
// and JavaScript data referenced by the links.
type Result struct {
	mu        sync.Mutex           // guards maps (but not their contents)
	status    string               // global analysis status
	fileInfos map[string]*fileInfo // keys are godoc file URLs
	pkgInfos  map[string]*pkgInfo  // keys are import paths
}

// fileInfo returns the fileInfo for the specified godoc file URL,
// constructing it as needed.  Thread-safe.
func (res *Result) fileInfo(url string) *fileInfo {
	res.mu.Lock()
	fi, ok := res.fileInfos[url]
	if !ok {
		if res.fileInfos == nil {
			res.fileInfos = make(map[string]*fileInfo)
		}
		fi = new(fileInfo)
		res.fileInfos[url] = fi
	}
	res.mu.Unlock()
	return fi
}

// Status returns a human-readable description of the current analysis status.
func (res *Result) Status() string {
	res.mu.Lock()
	defer res.mu.Unlock()
	return res.status
}

func (res *Result) setStatusf(format string, args ...interface{}) {
	res.mu.Lock()
	res.status = fmt.Sprintf(format, args...)
	log.Printf(format, args...)
	res.mu.Unlock()
}

// FileInfo returns new slices containing opaque JSON values and the
// HTML link markup for the specified godoc file URL.  Thread-safe.
// Callers must not mutate the elements.
// It returns "zero" if no data is available.
//
func (res *Result) FileInfo(url string) (fi FileInfo) {
	return res.fileInfo(url).get()
}

// pkgInfo returns the pkgInfo for the specified import path,
// constructing it as needed.  Thread-safe.
func (res *Result) pkgInfo(importPath string) *pkgInfo {
	res.mu.Lock()
	pi, ok := res.pkgInfos[importPath]
	if !ok {
		if res.pkgInfos == nil {
			res.pkgInfos = make(map[string]*pkgInfo)
		}
		pi = new(pkgInfo)
		res.pkgInfos[importPath] = pi
	}
	res.mu.Unlock()
	return pi
}

// PackageInfo returns new slices of JSON values for the callgraph and
// type info for the specified package.  Thread-safe.
// Callers must not mutate its fields.
// PackageInfo returns "zero" if no data is available.
//
func (res *Result) PackageInfo(importPath string) PackageInfo {
	return res.pkgInfo(importPath).get()
}

// -- analysis ---------------------------------------------------------

type analysis struct {
	result    *Result
	prog      *ssa.Program
	ops       []chanOp       // all channel ops in program
	allNamed  []*types.Named // all named types in the program
	ptaConfig pointer.Config
	path2url  map[string]string // maps openable path to godoc file URL (/src/fmt/print.go)
	pcgs      map[*ssa.Package]*packageCallGraph
}

// fileAndOffset returns the file and offset for a given pos.
func (a *analysis) fileAndOffset(pos token.Pos) (fi *fileInfo, offset int) {
	return a.fileAndOffsetPosn(a.prog.Fset.Position(pos))
}

// fileAndOffsetPosn returns the file and offset for a given position.
func (a *analysis) fileAndOffsetPosn(posn token.Position) (fi *fileInfo, offset int) {
	url := a.path2url[posn.Filename]
	return a.result.fileInfo(url), posn.Offset
}

// posURL returns the URL of the source extent [pos, pos+len).
func (a *analysis) posURL(pos token.Pos, len int) string {
	if pos == token.NoPos {
		return ""
	}
	posn := a.prog.Fset.Position(pos)
	url := a.path2url[posn.Filename]
	return fmt.Sprintf("%s?s=%d:%d#L%d",
		url, posn.Offset, posn.Offset+len, posn.Line)
}

// ----------------------------------------------------------------------

// Run runs program analysis and computes the resulting markup,
// populating *result in a thread-safe manner, first with type
// information then later with pointer analysis information if
// enabled by the pta flag.
//
func Run(pta bool, result *Result) {
	conf := loader.Config{
		AllowErrors: true,
	}

	// Silence the default error handler.
	// Don't print all errors; we'll report just
	// one per errant package later.
	conf.TypeChecker.Error = func(e error) {}

	var roots, args []string // roots[i] ends with os.PathSeparator

	// Enumerate packages in $GOROOT.
	root := filepath.Join(runtime.GOROOT(), "src") + string(os.PathSeparator)
	roots = append(roots, root)
	args = allPackages(root)
	log.Printf("GOROOT=%s: %s\n", root, args)

	// Enumerate packages in $GOPATH.
	for i, dir := range filepath.SplitList(build.Default.GOPATH) {
		root := filepath.Join(dir, "src") + string(os.PathSeparator)
		roots = append(roots, root)
		pkgs := allPackages(root)
		log.Printf("GOPATH[%d]=%s: %s\n", i, root, pkgs)
		args = append(args, pkgs...)
	}

	// Uncomment to make startup quicker during debugging.
	//args = []string{"golang.org/x/tools/cmd/godoc"}
	//args = []string{"fmt"}

	if _, err := conf.FromArgs(args, true); err != nil {
		// TODO(adonovan): degrade gracefully, not fail totally.
		// (The crippling case is a parse error in an external test file.)
		result.setStatusf("Analysis failed: %s.", err) // import error
		return
	}

	result.setStatusf("Loading and type-checking packages...")
	iprog, err := conf.Load()
	if iprog != nil {
		// Report only the first error of each package.
		for _, info := range iprog.AllPackages {
			for _, err := range info.Errors {
				fmt.Fprintln(os.Stderr, err)
				break
			}
		}
		log.Printf("Loaded %d packages.", len(iprog.AllPackages))
	}
	if err != nil {
		result.setStatusf("Loading failed: %s.\n", err)
		return
	}

	// Create SSA-form program representation.
	// Only the transitively error-free packages are used.
	prog := ssautil.CreateProgram(iprog, ssa.GlobalDebug)

	// Compute the set of main packages, including testmain.
	allPackages := prog.AllPackages()
	var mainPkgs []*ssa.Package
	if testmain := prog.CreateTestMainPackage(allPackages...); testmain != nil {
		mainPkgs = append(mainPkgs, testmain)
		if p := testmain.Const("packages"); p != nil {
			log.Printf("Tested packages: %v", exact.StringVal(p.Value.Value))
		}
	}
	for _, pkg := range allPackages {
		if pkg.Pkg.Name() == "main" && pkg.Func("main") != nil {
			mainPkgs = append(mainPkgs, pkg)
		}
	}
	log.Print("Transitively error-free main packages: ", mainPkgs)

	// Build SSA code for bodies of all functions in the whole program.
	result.setStatusf("Constructing SSA form...")
	prog.Build()
	log.Print("SSA construction complete")

	a := analysis{
		result: result,
		prog:   prog,
		pcgs:   make(map[*ssa.Package]*packageCallGraph),
	}

	// Build a mapping from openable filenames to godoc file URLs,
	// i.e. "/src/" plus path relative to GOROOT/src or GOPATH[i]/src.
	a.path2url = make(map[string]string)
	for _, info := range iprog.AllPackages {
	nextfile:
		for _, f := range info.Files {
			if f.Pos() == 0 {
				continue // e.g. files generated by cgo
			}
			abs := iprog.Fset.File(f.Pos()).Name()
			// Find the root to which this file belongs.
			for _, root := range roots {
				rel := strings.TrimPrefix(abs, root)
				if len(rel) < len(abs) {
					a.path2url[abs] = "/src/" + filepath.ToSlash(rel)
					continue nextfile
				}
			}

			log.Printf("Can't locate file %s (package %q) beneath any root",
				abs, info.Pkg.Path())
		}
	}

	// Add links for scanner, parser, type-checker errors.
	// TODO(adonovan): fix: these links can overlap with
	// identifier markup, causing the renderer to emit some
	// characters twice.
	errors := make(map[token.Position][]string)
	for _, info := range iprog.AllPackages {
		for _, err := range info.Errors {
			switch err := err.(type) {
			case types.Error:
				posn := a.prog.Fset.Position(err.Pos)
				errors[posn] = append(errors[posn], err.Msg)
			case scanner.ErrorList:
				for _, e := range err {
					errors[e.Pos] = append(errors[e.Pos], e.Msg)
				}
			default:
				log.Printf("Package %q has error (%T) without position: %v\n",
					info.Pkg.Path(), err, err)
			}
		}
	}
	for posn, errs := range errors {
		fi, offset := a.fileAndOffsetPosn(posn)
		fi.addLink(errorLink{
			start: offset,
			msg:   strings.Join(errs, "\n"),
		})
	}

	// ---------- type-based analyses ----------

	// Compute the all-pairs IMPLEMENTS relation.
	// Collect all named types, even local types
	// (which can have methods via promotion)
	// and the built-in "error".
	errorType := types.Universe.Lookup("error").Type().(*types.Named)
	a.allNamed = append(a.allNamed, errorType)
	for _, info := range iprog.AllPackages {
		for _, obj := range info.Defs {
			if obj, ok := obj.(*types.TypeName); ok {
				a.allNamed = append(a.allNamed, obj.Type().(*types.Named))
			}
		}
	}
	log.Print("Computing implements relation...")
	facts := computeImplements(&a.prog.MethodSets, a.allNamed)

	// Add the type-based analysis results.
	log.Print("Extracting type info...")
	for _, info := range iprog.AllPackages {
		a.doTypeInfo(info, facts)
	}

	a.visitInstrs(pta)

	result.setStatusf("Type analysis complete.")

	if pta {
		a.pointer(mainPkgs)
	}
}

// visitInstrs visits all SSA instructions in the program.
func (a *analysis) visitInstrs(pta bool) {
	log.Print("Visit instructions...")
	for fn := range ssautil.AllFunctions(a.prog) {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				// CALLEES (static)
				// (Dynamic calls require pointer analysis.)
				//
				// We use the SSA representation to find the static callee,
				// since in many cases it does better than the
				// types.Info.{Refs,Selection} information.  For example:
				//
				//   defer func(){}()      // static call to anon function
				//   f := func(){}; f()    // static call to anon function
				//   f := fmt.Println; f() // static call to named function
				//
				// The downside is that we get no static callee information
				// for packages that (transitively) contain errors.
				if site, ok := instr.(ssa.CallInstruction); ok {
					if callee := site.Common().StaticCallee(); callee != nil {
						// TODO(adonovan): callgraph: elide wrappers.
						// (Do static calls ever go to wrappers?)
						if site.Common().Pos() != token.NoPos {
							a.addCallees(site, []*ssa.Function{callee})
						}
					}
				}

				if !pta {
					continue
				}

				// CHANNEL PEERS
				// Collect send/receive/close instructions in the whole ssa.Program.
				for _, op := range chanOps(instr) {
					a.ops = append(a.ops, op)
					a.ptaConfig.AddQuery(op.ch) // add channel ssa.Value to PTA query
				}
			}
		}
	}
	log.Print("Visit instructions complete")
}

// pointer runs the pointer analysis.
func (a *analysis) pointer(mainPkgs []*ssa.Package) {
	// Run the pointer analysis and build the complete callgraph.
	a.ptaConfig.Mains = mainPkgs
	a.ptaConfig.BuildCallGraph = true
	a.ptaConfig.Reflection = false // (for now)

	a.result.setStatusf("Pointer analysis running...")

	ptares, err := pointer.Analyze(&a.ptaConfig)
	if err != nil {
		// If this happens, it indicates a bug.
		a.result.setStatusf("Pointer analysis failed: %s.", err)
		return
	}
	log.Print("Pointer analysis complete.")

	// Add the results of pointer analysis.

	a.result.setStatusf("Computing channel peers...")
	a.doChannelPeers(ptares.Queries)
	a.result.setStatusf("Computing dynamic call graph edges...")
	a.doCallgraph(ptares.CallGraph)

	a.result.setStatusf("Analysis complete.")
}

type linksByStart []Link

func (a linksByStart) Less(i, j int) bool { return a[i].Start() < a[j].Start() }
func (a linksByStart) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a linksByStart) Len() int           { return len(a) }

// allPackages returns a new sorted slice of all packages beneath the
// specified package root directory, e.g. $GOROOT/src or $GOPATH/src.
// Derived from from go/ssa/stdlib_test.go
// root must end with os.PathSeparator.
//
// TODO(adonovan): use buildutil.AllPackages when the tree thaws.
func allPackages(root string) []string {
	var pkgs []string
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if info == nil {
			return nil // non-existent root directory?
		}
		if !info.IsDir() {
			return nil // not a directory
		}
		// Prune the search if we encounter any of these names:
		base := filepath.Base(path)
		if base == "testdata" || strings.HasPrefix(base, ".") {
			return filepath.SkipDir
		}
		pkg := filepath.ToSlash(strings.TrimPrefix(path, root))
		switch pkg {
		case "builtin":
			return filepath.SkipDir
		case "":
			return nil // ignore root of tree
		}
		pkgs = append(pkgs, pkg)
		return nil
	})
	return pkgs
}
