// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"cmd/internal/cov/covcmd"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"internal/coverage"
	"internal/coverage/encodemeta"
	"internal/coverage/slicewriter"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"cmd/internal/edit"
	"cmd/internal/objabi"
)

const usageMessage = "" +
	`Usage of 'go tool cover':
Given a coverage profile produced by 'go test':
	go test -coverprofile=c.out

Open a web browser displaying annotated source code:
	go tool cover -html=c.out

Write out an HTML file instead of launching a web browser:
	go tool cover -html=c.out -o coverage.html

Display coverage percentages to stdout for each function:
	go tool cover -func=c.out

Finally, to generate modified source code with coverage annotations
for a package (what go test -cover does):
	go tool cover -mode=set -var=CoverageVariableName \
		-pkgcfg=<config> -outfilelist=<file> file1.go ... fileN.go

where -pkgcfg points to a file containing the package path,
package name, module path, and related info from "go build",
and -outfilelist points to a file containing the filenames
of the instrumented output files (one per input file).
See https://pkg.go.dev/cmd/internal/cov/covcmd#CoverPkgConfig for
more on the package config.
`

func usage() {
	fmt.Fprint(os.Stderr, usageMessage)
	fmt.Fprintln(os.Stderr, "\nFlags:")
	flag.PrintDefaults()
	fmt.Fprintln(os.Stderr, "\n  Only one of -html, -func, or -mode may be set.")
	os.Exit(2)
}

var (
	mode             = flag.String("mode", "", "coverage mode: set, count, atomic")
	varVar           = flag.String("var", "GoCover", "name of coverage variable to generate")
	output           = flag.String("o", "", "file for output")
	outfilelist      = flag.String("outfilelist", "", "file containing list of output files (one per line) if -pkgcfg is in use")
	htmlOut          = flag.String("html", "", "generate HTML representation of coverage profile")
	css              = flag.String("css", "", "CSS file for HTML representation of coverage profile")
	funcOut          = flag.String("func", "", "output coverage profile information for each function")
	pkgcfg           = flag.String("pkgcfg", "", "enable full-package instrumentation mode using params from specified config file")
	pkgconfig        covcmd.CoverPkgConfig
	outputfiles      []string // list of *.cover.go instrumented outputs to write, one per input (set when -pkgcfg is in use)
	profile          string   // The profile to read; the value of -html or -func
	counterStmt      func(*File, string) string
	covervarsoutfile string // an additional Go source file into which we'll write definitions of coverage counter variables + meta data variables (set when -pkgcfg is in use).
	cmode            coverage.CounterMode
	cgran            coverage.CounterGranularity
)

const (
	atomicPackagePath = "sync/atomic"
	atomicPackageName = "_cover_atomic_"
)

func main() {
	objabi.AddVersionFlag()
	flag.Usage = usage
	objabi.Flagparse(usage)

	// Usage information when no arguments.
	if flag.NFlag() == 0 && flag.NArg() == 0 {
		flag.Usage()
	}

	err := parseFlags()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, `For usage information, run "go tool cover -help"`)
		os.Exit(2)
	}

	// Generate coverage-annotated source.
	if *mode != "" {
		annotate(flag.Args())
		return
	}

	// Output HTML or function coverage information.
	if *htmlOut != "" {
		err = htmlOutput(profile, *css, *output)
	} else {
		err = funcOutput(profile, *output)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "cover: %v\n", err)
		os.Exit(2)
	}
}

// parseFlags sets the profile and counterStmt globals and performs validations.
func parseFlags() error {
	profile = *htmlOut
	if *funcOut != "" {
		if profile != "" {
			return fmt.Errorf("too many options")
		}
		profile = *funcOut
	}

	if *css != "" && *htmlOut == "" {
		return fmt.Errorf("-css option requires -html")
	}

	// Must either display a profile or rewrite Go source.
	if (profile == "") == (*mode == "") {
		return fmt.Errorf("too many options")
	}

	if *varVar != "" && !token.IsIdentifier(*varVar) {
		return fmt.Errorf("-var: %q is not a valid identifier", *varVar)
	}

	if *mode != "" {
		switch *mode {
		case "set":
			counterStmt = setCounterStmt
			cmode = coverage.CtrModeSet
		case "count":
			counterStmt = incCounterStmt
			cmode = coverage.CtrModeCount
		case "atomic":
			counterStmt = atomicCounterStmt
			cmode = coverage.CtrModeAtomic
		case "regonly":
			counterStmt = nil
			cmode = coverage.CtrModeRegOnly
		case "testmain":
			counterStmt = nil
			cmode = coverage.CtrModeTestMain
		default:
			return fmt.Errorf("unknown -mode %v", *mode)
		}

		if flag.NArg() == 0 {
			return fmt.Errorf("missing source file(s)")
		} else {
			if *pkgcfg != "" {
				if *output != "" {
					return fmt.Errorf("please use '-outfilelist' flag instead of '-o'")
				}
				var err error
				if outputfiles, err = readOutFileList(*outfilelist); err != nil {
					return err
				}
				covervarsoutfile = outputfiles[0]
				outputfiles = outputfiles[1:]
				numInputs := len(flag.Args())
				numOutputs := len(outputfiles)
				if numOutputs != numInputs {
					return fmt.Errorf("number of output files (%d) not equal to number of input files (%d)", numOutputs, numInputs)
				}
				if err := readPackageConfig(*pkgcfg); err != nil {
					return err
				}
				return nil
			} else {
				if *outfilelist != "" {
					return fmt.Errorf("'-outfilelist' flag applicable only when -pkgcfg used")
				}
			}
			if flag.NArg() == 1 {
				return nil
			}
		}
	} else if flag.NArg() == 0 {
		return nil
	}
	return fmt.Errorf("too many arguments")
}

func readOutFileList(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error reading -outfilelist file %q: %v", path, err)
	}
	return strings.Split(strings.TrimSpace(string(data)), "\n"), nil
}

func readPackageConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("error reading pkgconfig file %q: %v", path, err)
	}
	if err := json.Unmarshal(data, &pkgconfig); err != nil {
		return fmt.Errorf("error reading pkgconfig file %q: %v", path, err)
	}
	switch pkgconfig.Granularity {
	case "perblock":
		cgran = coverage.CtrGranularityPerBlock
	case "perfunc":
		cgran = coverage.CtrGranularityPerFunc
	default:
		return fmt.Errorf(`%s: pkgconfig requires perblock/perfunc value`, path)
	}
	return nil
}

// Block represents the information about a basic block to be recorded in the analysis.
// Note: Our definition of basic block is based on control structures; we don't break
// apart && and ||. We could but it doesn't seem important enough to bother.
type Block struct {
	startByte token.Pos
	endByte   token.Pos
	numStmt   int
}

// Package holds package-specific state.
type Package struct {
	mdb            *encodemeta.CoverageMetaDataBuilder
	counterLengths []int
}

// Function holds func-specific state.
type Func struct {
	units      []coverage.CoverableUnit
	counterVar string
}

// File is a wrapper for the state of a file used in the parser.
// The basic parse tree walker is a method of this type.
type File struct {
	fset    *token.FileSet
	name    string // Name of file.
	astFile *ast.File
	blocks  []Block
	content []byte
	edit    *edit.Buffer
	mdb     *encodemeta.CoverageMetaDataBuilder
	fn      Func
	pkg     *Package
}

// findText finds text in the original source, starting at pos.
// It correctly skips over comments and assumes it need not
// handle quoted strings.
// It returns a byte offset within f.src.
func (f *File) findText(pos token.Pos, text string) int {
	b := []byte(text)
	start := f.offset(pos)
	i := start
	s := f.content
	for i < len(s) {
		if bytes.HasPrefix(s[i:], b) {
			return i
		}
		if i+2 <= len(s) && s[i] == '/' && s[i+1] == '/' {
			for i < len(s) && s[i] != '\n' {
				i++
			}
			continue
		}
		if i+2 <= len(s) && s[i] == '/' && s[i+1] == '*' {
			for i += 2; ; i++ {
				if i+2 > len(s) {
					return 0
				}
				if s[i] == '*' && s[i+1] == '/' {
					i += 2
					break
				}
			}
			continue
		}
		i++
	}
	return -1
}

// Visit implements the ast.Visitor interface.
func (f *File) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.BlockStmt:
		// If it's a switch or select, the body is a list of case clauses; don't tag the block itself.
		if len(n.List) > 0 {
			switch n.List[0].(type) {
			case *ast.CaseClause: // switch
				for _, n := range n.List {
					clause := n.(*ast.CaseClause)
					f.addCounters(clause.Colon+1, clause.Colon+1, clause.End(), clause.Body, false)
				}
				return f
			case *ast.CommClause: // select
				for _, n := range n.List {
					clause := n.(*ast.CommClause)
					f.addCounters(clause.Colon+1, clause.Colon+1, clause.End(), clause.Body, false)
				}
				return f
			}
		}
		f.addCounters(n.Lbrace, n.Lbrace+1, n.Rbrace+1, n.List, true) // +1 to step past closing brace.
	case *ast.IfStmt:
		if n.Init != nil {
			ast.Walk(f, n.Init)
		}
		ast.Walk(f, n.Cond)
		ast.Walk(f, n.Body)
		if n.Else == nil {
			return nil
		}
		// The elses are special, because if we have
		//	if x {
		//	} else if y {
		//	}
		// we want to cover the "if y". To do this, we need a place to drop the counter,
		// so we add a hidden block:
		//	if x {
		//	} else {
		//		if y {
		//		}
		//	}
		elseOffset := f.findText(n.Body.End(), "else")
		if elseOffset < 0 {
			panic("lost else")
		}
		f.edit.Insert(elseOffset+4, "{")
		f.edit.Insert(f.offset(n.Else.End()), "}")

		// We just created a block, now walk it.
		// Adjust the position of the new block to start after
		// the "else". That will cause it to follow the "{"
		// we inserted above.
		pos := f.fset.File(n.Body.End()).Pos(elseOffset + 4)
		switch stmt := n.Else.(type) {
		case *ast.IfStmt:
			block := &ast.BlockStmt{
				Lbrace: pos,
				List:   []ast.Stmt{stmt},
				Rbrace: stmt.End(),
			}
			n.Else = block
		case *ast.BlockStmt:
			stmt.Lbrace = pos
		default:
			panic("unexpected node type in if")
		}
		ast.Walk(f, n.Else)
		return nil
	case *ast.SelectStmt:
		// Don't annotate an empty select - creates a syntax error.
		if n.Body == nil || len(n.Body.List) == 0 {
			return nil
		}
	case *ast.SwitchStmt:
		// Don't annotate an empty switch - creates a syntax error.
		if n.Body == nil || len(n.Body.List) == 0 {
			if n.Init != nil {
				ast.Walk(f, n.Init)
			}
			if n.Tag != nil {
				ast.Walk(f, n.Tag)
			}
			return nil
		}
	case *ast.TypeSwitchStmt:
		// Don't annotate an empty type switch - creates a syntax error.
		if n.Body == nil || len(n.Body.List) == 0 {
			if n.Init != nil {
				ast.Walk(f, n.Init)
			}
			ast.Walk(f, n.Assign)
			return nil
		}
	case *ast.FuncDecl:
		// Don't annotate functions with blank names - they cannot be executed.
		// Similarly for bodyless funcs.
		if n.Name.Name == "_" || n.Body == nil {
			return nil
		}
		fname := n.Name.Name
		// Skip AddUint32 and StoreUint32 if we're instrumenting
		// sync/atomic itself in atomic mode (out of an abundance of
		// caution), since as part of the instrumentation process we
		// add calls to AddUint32/StoreUint32, and we don't want to
		// somehow create an infinite loop.
		//
		// Note that in the current implementation (Go 1.20) both
		// routines are assembly stubs that forward calls to the
		// runtime/internal/atomic equivalents, hence the infinite
		// loop scenario is purely theoretical (maybe if in some
		// future implementation one of these functions might be
		// written in Go). See #57445 for more details.
		if atomicOnAtomic() && (fname == "AddUint32" || fname == "StoreUint32") {
			return nil
		}
		// Determine proper function or method name.
		if r := n.Recv; r != nil && len(r.List) == 1 {
			t := r.List[0].Type
			star := ""
			if p, _ := t.(*ast.StarExpr); p != nil {
				t = p.X
				star = "*"
			}
			if p, _ := t.(*ast.Ident); p != nil {
				fname = star + p.Name + "." + fname
			}
		}
		walkBody := true
		if *pkgcfg != "" {
			f.preFunc(n, fname)
			if pkgconfig.Granularity == "perfunc" {
				walkBody = false
			}
		}
		if walkBody {
			ast.Walk(f, n.Body)
		}
		if *pkgcfg != "" {
			flit := false
			f.postFunc(n, fname, flit, n.Body)
		}
		return nil
	case *ast.FuncLit:
		// For function literals enclosed in functions, just glom the
		// code for the literal in with the enclosing function (for now).
		if f.fn.counterVar != "" {
			return f
		}

		// Hack: function literals aren't named in the go/ast representation,
		// and we don't know what name the compiler will choose. For now,
		// just make up a descriptive name.
		pos := n.Pos()
		p := f.fset.File(pos).Position(pos)
		fname := fmt.Sprintf("func.L%d.C%d", p.Line, p.Column)
		if *pkgcfg != "" {
			f.preFunc(n, fname)
		}
		if pkgconfig.Granularity != "perfunc" {
			ast.Walk(f, n.Body)
		}
		if *pkgcfg != "" {
			flit := true
			f.postFunc(n, fname, flit, n.Body)
		}
		return nil
	}
	return f
}

func mkCounterVarName(idx int) string {
	return fmt.Sprintf("%s_%d", *varVar, idx)
}

func mkPackageIdVar() string {
	return *varVar + "P"
}

func mkMetaVar() string {
	return *varVar + "M"
}

func mkPackageIdExpression() string {
	ppath := pkgconfig.PkgPath
	if hcid := coverage.HardCodedPkgID(ppath); hcid != -1 {
		return fmt.Sprintf("uint32(%d)", uint32(hcid))
	}
	return mkPackageIdVar()
}

func (f *File) preFunc(fn ast.Node, fname string) {
	f.fn.units = f.fn.units[:0]

	// create a new counter variable for this function.
	cv := mkCounterVarName(len(f.pkg.counterLengths))
	f.fn.counterVar = cv
}

func (f *File) postFunc(fn ast.Node, funcname string, flit bool, body *ast.BlockStmt) {

	// Tack on single counter write if we are in "perfunc" mode.
	singleCtr := ""
	if pkgconfig.Granularity == "perfunc" {
		singleCtr = "; " + f.newCounter(fn.Pos(), fn.Pos(), 1)
	}

	// record the length of the counter var required.
	nc := len(f.fn.units) + coverage.FirstCtrOffset
	f.pkg.counterLengths = append(f.pkg.counterLengths, nc)

	// FIXME: for windows, do we want "\" and not "/"? Need to test here.
	// Currently filename is formed as packagepath + "/" + basename.
	fnpos := f.fset.Position(fn.Pos())
	ppath := pkgconfig.PkgPath
	filename := ppath + "/" + filepath.Base(fnpos.Filename)

	// The convention for cmd/cover is that if the go command that
	// kicks off coverage specifies a local import path (e.g. "go test
	// -cover ./thispackage"), the tool will capture full pathnames
	// for source files instead of relative paths, which tend to work
	// more smoothly for "go tool cover -html". See also issue #56433
	// for more details.
	if pkgconfig.Local {
		filename = f.name
	}

	// Hand off function to meta-data builder.
	fd := coverage.FuncDesc{
		Funcname: funcname,
		Srcfile:  filename,
		Units:    f.fn.units,
		Lit:      flit,
	}
	funcId := f.mdb.AddFunc(fd)

	hookWrite := func(cv string, which int, val string) string {
		return fmt.Sprintf("%s[%d] = %s", cv, which, val)
	}
	if *mode == "atomic" {
		hookWrite = func(cv string, which int, val string) string {
			return fmt.Sprintf("%sStoreUint32(&%s[%d], %s)",
				atomicPackagePrefix(), cv, which, val)
		}
	}

	// Generate the registration hook sequence for the function. This
	// sequence looks like
	//
	//   counterVar[0] = <num_units>
	//   counterVar[1] = pkgId
	//   counterVar[2] = fnId
	//
	cv := f.fn.counterVar
	regHook := hookWrite(cv, 0, strconv.Itoa(len(f.fn.units))) + " ; " +
		hookWrite(cv, 1, mkPackageIdExpression()) + " ; " +
		hookWrite(cv, 2, strconv.Itoa(int(funcId))) + singleCtr

	// Insert the registration sequence into the function. We want this sequence to
	// appear before any counter updates, so use a hack to ensure that this edit
	// applies before the edit corresponding to the prolog counter update.

	boff := f.offset(body.Pos())
	ipos := f.fset.File(body.Pos()).Pos(boff)
	ip := f.offset(ipos)
	f.edit.Replace(ip, ip+1, string(f.content[ipos-1])+regHook+" ; ")

	f.fn.counterVar = ""
}

func annotate(names []string) {
	var p *Package
	if *pkgcfg != "" {
		pp := pkgconfig.PkgPath
		pn := pkgconfig.PkgName
		mp := pkgconfig.ModulePath
		mdb, err := encodemeta.NewCoverageMetaDataBuilder(pp, pn, mp)
		if err != nil {
			log.Fatalf("creating coverage meta-data builder: %v\n", err)
		}
		p = &Package{
			mdb: mdb,
		}
	}
	// TODO: process files in parallel here if it matters.
	for k, name := range names {
		if strings.ContainsAny(name, "\r\n") {
			// annotateFile uses '//line' directives, which don't permit newlines.
			log.Fatalf("cover: input path contains newline character: %q", name)
		}

		fd := os.Stdout
		isStdout := true
		if *pkgcfg != "" {
			var err error
			fd, err = os.Create(outputfiles[k])
			if err != nil {
				log.Fatalf("cover: %s", err)
			}
			isStdout = false
		} else if *output != "" {
			var err error
			fd, err = os.Create(*output)
			if err != nil {
				log.Fatalf("cover: %s", err)
			}
			isStdout = false
		}
		p.annotateFile(name, fd)
		if !isStdout {
			if err := fd.Close(); err != nil {
				log.Fatalf("cover: %s", err)
			}
		}
	}

	if *pkgcfg != "" {
		fd, err := os.Create(covervarsoutfile)
		if err != nil {
			log.Fatalf("cover: %s", err)
		}
		p.emitMetaData(fd)
		if err := fd.Close(); err != nil {
			log.Fatalf("cover: %s", err)
		}
	}
}

func (p *Package) annotateFile(name string, fd io.Writer) {
	fset := token.NewFileSet()
	content, err := os.ReadFile(name)
	if err != nil {
		log.Fatalf("cover: %s: %s", name, err)
	}
	parsedFile, err := parser.ParseFile(fset, name, content, parser.ParseComments)
	if err != nil {
		log.Fatalf("cover: %s: %s", name, err)
	}

	file := &File{
		fset:    fset,
		name:    name,
		content: content,
		edit:    edit.NewBuffer(content),
		astFile: parsedFile,
	}
	if p != nil {
		file.mdb = p.mdb
		file.pkg = p
	}

	if *mode == "atomic" {
		// Add import of sync/atomic immediately after package clause.
		// We do this even if there is an existing import, because the
		// existing import may be shadowed at any given place we want
		// to refer to it, and our name (_cover_atomic_) is less likely to
		// be shadowed. The one exception is if we're visiting the
		// sync/atomic package itself, in which case we can refer to
		// functions directly without an import prefix. See also #57445.
		if pkgconfig.PkgPath != "sync/atomic" {
			file.edit.Insert(file.offset(file.astFile.Name.End()),
				fmt.Sprintf("; import %s %q", atomicPackageName, atomicPackagePath))
		}
	}
	if pkgconfig.PkgName == "main" {
		file.edit.Insert(file.offset(file.astFile.Name.End()),
			"; import _ \"runtime/coverage\"")
	}

	if counterStmt != nil {
		ast.Walk(file, file.astFile)
	}
	newContent := file.edit.Bytes()

	if strings.ContainsAny(name, "\r\n") {
		// This should have been checked by the caller already, but we double check
		// here just to be sure we haven't missed a caller somewhere.
		panic(fmt.Sprintf("annotateFile: name contains unexpected newline character: %q", name))
	}
	fmt.Fprintf(fd, "//line %s:1:1\n", name)
	fd.Write(newContent)

	// After printing the source tree, add some declarations for the
	// counters etc. We could do this by adding to the tree, but it's
	// easier just to print the text.
	file.addVariables(fd)

	// Emit a reference to the atomic package to avoid
	// import and not used error when there's no code in a file.
	if *mode == "atomic" {
		fmt.Fprintf(fd, "\nvar _ = %sLoadUint32\n", atomicPackagePrefix())
	}
}

// setCounterStmt returns the expression: __count[23] = 1.
func setCounterStmt(f *File, counter string) string {
	return fmt.Sprintf("%s = 1", counter)
}

// incCounterStmt returns the expression: __count[23]++.
func incCounterStmt(f *File, counter string) string {
	return fmt.Sprintf("%s++", counter)
}

// atomicCounterStmt returns the expression: atomic.AddUint32(&__count[23], 1)
func atomicCounterStmt(f *File, counter string) string {
	return fmt.Sprintf("%sAddUint32(&%s, 1)", atomicPackagePrefix(), counter)
}

// newCounter creates a new counter expression of the appropriate form.
func (f *File) newCounter(start, end token.Pos, numStmt int) string {
	var stmt string
	if *pkgcfg != "" {
		slot := len(f.fn.units) + coverage.FirstCtrOffset
		if f.fn.counterVar == "" {
			panic("internal error: counter var unset")
		}
		stmt = counterStmt(f, fmt.Sprintf("%s[%d]", f.fn.counterVar, slot))
		stpos := f.fset.Position(start)
		enpos := f.fset.Position(end)
		stpos, enpos = dedup(stpos, enpos)
		unit := coverage.CoverableUnit{
			StLine:  uint32(stpos.Line),
			StCol:   uint32(stpos.Column),
			EnLine:  uint32(enpos.Line),
			EnCol:   uint32(enpos.Column),
			NxStmts: uint32(numStmt),
		}
		f.fn.units = append(f.fn.units, unit)
	} else {
		stmt = counterStmt(f, fmt.Sprintf("%s.Count[%d]", *varVar,
			len(f.blocks)))
		f.blocks = append(f.blocks, Block{start, end, numStmt})
	}
	return stmt
}

// addCounters takes a list of statements and adds counters to the beginning of
// each basic block at the top level of that list. For instance, given
//
//	S1
//	if cond {
//		S2
//	}
//	S3
//
// counters will be added before S1 and before S3. The block containing S2
// will be visited in a separate call.
// TODO: Nested simple blocks get unnecessary (but correct) counters
func (f *File) addCounters(pos, insertPos, blockEnd token.Pos, list []ast.Stmt, extendToClosingBrace bool) {
	// Special case: make sure we add a counter to an empty block. Can't do this below
	// or we will add a counter to an empty statement list after, say, a return statement.
	if len(list) == 0 {
		f.edit.Insert(f.offset(insertPos), f.newCounter(insertPos, blockEnd, 0)+";")
		return
	}
	// Make a copy of the list, as we may mutate it and should leave the
	// existing list intact.
	list = append([]ast.Stmt(nil), list...)
	// We have a block (statement list), but it may have several basic blocks due to the
	// appearance of statements that affect the flow of control.
	for {
		// Find first statement that affects flow of control (break, continue, if, etc.).
		// It will be the last statement of this basic block.
		var last int
		end := blockEnd
		for last = 0; last < len(list); last++ {
			stmt := list[last]
			end = f.statementBoundary(stmt)
			if f.endsBasicSourceBlock(stmt) {
				// If it is a labeled statement, we need to place a counter between
				// the label and its statement because it may be the target of a goto
				// and thus start a basic block. That is, given
				//	foo: stmt
				// we need to create
				//	foo: ; stmt
				// and mark the label as a block-terminating statement.
				// The result will then be
				//	foo: COUNTER[n]++; stmt
				// However, we can't do this if the labeled statement is already
				// a control statement, such as a labeled for.
				if label, isLabel := stmt.(*ast.LabeledStmt); isLabel && !f.isControl(label.Stmt) {
					newLabel := *label
					newLabel.Stmt = &ast.EmptyStmt{
						Semicolon: label.Stmt.Pos(),
						Implicit:  true,
					}
					end = label.Pos() // Previous block ends before the label.
					list[last] = &newLabel
					// Open a gap and drop in the old statement, now without a label.
					list = append(list, nil)
					copy(list[last+1:], list[last:])
					list[last+1] = label.Stmt
				}
				last++
				extendToClosingBrace = false // Block is broken up now.
				break
			}
		}
		if extendToClosingBrace {
			end = blockEnd
		}
		if pos != end { // Can have no source to cover if e.g. blocks abut.
			f.edit.Insert(f.offset(insertPos), f.newCounter(pos, end, last)+";")
		}
		list = list[last:]
		if len(list) == 0 {
			break
		}
		pos = list[0].Pos()
		insertPos = pos
	}
}

// hasFuncLiteral reports the existence and position of the first func literal
// in the node, if any. If a func literal appears, it usually marks the termination
// of a basic block because the function body is itself a block.
// Therefore we draw a line at the start of the body of the first function literal we find.
// TODO: what if there's more than one? Probably doesn't matter much.
func hasFuncLiteral(n ast.Node) (bool, token.Pos) {
	if n == nil {
		return false, 0
	}
	var literal funcLitFinder
	ast.Walk(&literal, n)
	return literal.found(), token.Pos(literal)
}

// statementBoundary finds the location in s that terminates the current basic
// block in the source.
func (f *File) statementBoundary(s ast.Stmt) token.Pos {
	// Control flow statements are easy.
	switch s := s.(type) {
	case *ast.BlockStmt:
		// Treat blocks like basic blocks to avoid overlapping counters.
		return s.Lbrace
	case *ast.IfStmt:
		found, pos := hasFuncLiteral(s.Init)
		if found {
			return pos
		}
		found, pos = hasFuncLiteral(s.Cond)
		if found {
			return pos
		}
		return s.Body.Lbrace
	case *ast.ForStmt:
		found, pos := hasFuncLiteral(s.Init)
		if found {
			return pos
		}
		found, pos = hasFuncLiteral(s.Cond)
		if found {
			return pos
		}
		found, pos = hasFuncLiteral(s.Post)
		if found {
			return pos
		}
		return s.Body.Lbrace
	case *ast.LabeledStmt:
		return f.statementBoundary(s.Stmt)
	case *ast.RangeStmt:
		found, pos := hasFuncLiteral(s.X)
		if found {
			return pos
		}
		return s.Body.Lbrace
	case *ast.SwitchStmt:
		found, pos := hasFuncLiteral(s.Init)
		if found {
			return pos
		}
		found, pos = hasFuncLiteral(s.Tag)
		if found {
			return pos
		}
		return s.Body.Lbrace
	case *ast.SelectStmt:
		return s.Body.Lbrace
	case *ast.TypeSwitchStmt:
		found, pos := hasFuncLiteral(s.Init)
		if found {
			return pos
		}
		return s.Body.Lbrace
	}
	// If not a control flow statement, it is a declaration, expression, call, etc. and it may have a function literal.
	// If it does, that's tricky because we want to exclude the body of the function from this block.
	// Draw a line at the start of the body of the first function literal we find.
	// TODO: what if there's more than one? Probably doesn't matter much.
	found, pos := hasFuncLiteral(s)
	if found {
		return pos
	}
	return s.End()
}

// endsBasicSourceBlock reports whether s changes the flow of control: break, if, etc.,
// or if it's just problematic, for instance contains a function literal, which will complicate
// accounting due to the block-within-an expression.
func (f *File) endsBasicSourceBlock(s ast.Stmt) bool {
	switch s := s.(type) {
	case *ast.BlockStmt:
		// Treat blocks like basic blocks to avoid overlapping counters.
		return true
	case *ast.BranchStmt:
		return true
	case *ast.ForStmt:
		return true
	case *ast.IfStmt:
		return true
	case *ast.LabeledStmt:
		return true // A goto may branch here, starting a new basic block.
	case *ast.RangeStmt:
		return true
	case *ast.SwitchStmt:
		return true
	case *ast.SelectStmt:
		return true
	case *ast.TypeSwitchStmt:
		return true
	case *ast.ExprStmt:
		// Calls to panic change the flow.
		// We really should verify that "panic" is the predefined function,
		// but without type checking we can't and the likelihood of it being
		// an actual problem is vanishingly small.
		if call, ok := s.X.(*ast.CallExpr); ok {
			if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "panic" && len(call.Args) == 1 {
				return true
			}
		}
	}
	found, _ := hasFuncLiteral(s)
	return found
}

// isControl reports whether s is a control statement that, if labeled, cannot be
// separated from its label.
func (f *File) isControl(s ast.Stmt) bool {
	switch s.(type) {
	case *ast.ForStmt, *ast.RangeStmt, *ast.SwitchStmt, *ast.SelectStmt, *ast.TypeSwitchStmt:
		return true
	}
	return false
}

// funcLitFinder implements the ast.Visitor pattern to find the location of any
// function literal in a subtree.
type funcLitFinder token.Pos

func (f *funcLitFinder) Visit(node ast.Node) (w ast.Visitor) {
	if f.found() {
		return nil // Prune search.
	}
	switch n := node.(type) {
	case *ast.FuncLit:
		*f = funcLitFinder(n.Body.Lbrace)
		return nil // Prune search.
	}
	return f
}

func (f *funcLitFinder) found() bool {
	return token.Pos(*f) != token.NoPos
}

// Sort interface for []block1; used for self-check in addVariables.

type block1 struct {
	Block
	index int
}

type blockSlice []block1

func (b blockSlice) Len() int           { return len(b) }
func (b blockSlice) Less(i, j int) bool { return b[i].startByte < b[j].startByte }
func (b blockSlice) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }

// offset translates a token position into a 0-indexed byte offset.
func (f *File) offset(pos token.Pos) int {
	return f.fset.Position(pos).Offset
}

// addVariables adds to the end of the file the declarations to set up the counter and position variables.
func (f *File) addVariables(w io.Writer) {
	if *pkgcfg != "" {
		return
	}
	// Self-check: Verify that the instrumented basic blocks are disjoint.
	t := make([]block1, len(f.blocks))
	for i := range f.blocks {
		t[i].Block = f.blocks[i]
		t[i].index = i
	}
	sort.Sort(blockSlice(t))
	for i := 1; i < len(t); i++ {
		if t[i-1].endByte > t[i].startByte {
			fmt.Fprintf(os.Stderr, "cover: internal error: block %d overlaps block %d\n", t[i-1].index, t[i].index)
			// Note: error message is in byte positions, not token positions.
			fmt.Fprintf(os.Stderr, "\t%s:#%d,#%d %s:#%d,#%d\n",
				f.name, f.offset(t[i-1].startByte), f.offset(t[i-1].endByte),
				f.name, f.offset(t[i].startByte), f.offset(t[i].endByte))
		}
	}

	// Declare the coverage struct as a package-level variable.
	fmt.Fprintf(w, "\nvar %s = struct {\n", *varVar)
	fmt.Fprintf(w, "\tCount     [%d]uint32\n", len(f.blocks))
	fmt.Fprintf(w, "\tPos       [3 * %d]uint32\n", len(f.blocks))
	fmt.Fprintf(w, "\tNumStmt   [%d]uint16\n", len(f.blocks))
	fmt.Fprintf(w, "} {\n")

	// Initialize the position array field.
	fmt.Fprintf(w, "\tPos: [3 * %d]uint32{\n", len(f.blocks))

	// A nice long list of positions. Each position is encoded as follows to reduce size:
	// - 32-bit starting line number
	// - 32-bit ending line number
	// - (16 bit ending column number << 16) | (16-bit starting column number).
	for i, block := range f.blocks {
		start := f.fset.Position(block.startByte)
		end := f.fset.Position(block.endByte)

		start, end = dedup(start, end)

		fmt.Fprintf(w, "\t\t%d, %d, %#x, // [%d]\n", start.Line, end.Line, (end.Column&0xFFFF)<<16|(start.Column&0xFFFF), i)
	}

	// Close the position array.
	fmt.Fprintf(w, "\t},\n")

	// Initialize the position array field.
	fmt.Fprintf(w, "\tNumStmt: [%d]uint16{\n", len(f.blocks))

	// A nice long list of statements-per-block, so we can give a conventional
	// valuation of "percent covered". To save space, it's a 16-bit number, so we
	// clamp it if it overflows - won't matter in practice.
	for i, block := range f.blocks {
		n := block.numStmt
		if n > 1<<16-1 {
			n = 1<<16 - 1
		}
		fmt.Fprintf(w, "\t\t%d, // %d\n", n, i)
	}

	// Close the statements-per-block array.
	fmt.Fprintf(w, "\t},\n")

	// Close the struct initialization.
	fmt.Fprintf(w, "}\n")
}

// It is possible for positions to repeat when there is a line
// directive that does not specify column information and the input
// has not been passed through gofmt.
// See issues #27530 and #30746.
// Tests are TestHtmlUnformatted and TestLineDup.
// We use a map to avoid duplicates.

// pos2 is a pair of token.Position values, used as a map key type.
type pos2 struct {
	p1, p2 token.Position
}

// seenPos2 tracks whether we have seen a token.Position pair.
var seenPos2 = make(map[pos2]bool)

// dedup takes a token.Position pair and returns a pair that does not
// duplicate any existing pair. The returned pair will have the Offset
// fields cleared.
func dedup(p1, p2 token.Position) (r1, r2 token.Position) {
	key := pos2{
		p1: p1,
		p2: p2,
	}

	// We want to ignore the Offset fields in the map,
	// since cover uses only file/line/column.
	key.p1.Offset = 0
	key.p2.Offset = 0

	for seenPos2[key] {
		key.p2.Column++
	}
	seenPos2[key] = true

	return key.p1, key.p2
}

func (p *Package) emitMetaData(w io.Writer) {
	if *pkgcfg == "" {
		return
	}

	// If the "EmitMetaFile" path has been set, invoke a helper
	// that will write out a pre-cooked meta-data file for this package
	// to the specified location, in effect simulating the execution
	// of a test binary that doesn't do any testing to speak of.
	if pkgconfig.EmitMetaFile != "" {
		p.emitMetaFile(pkgconfig.EmitMetaFile)
	}

	// Something went wrong if regonly/testmain mode is in effect and
	// we have instrumented functions.
	if counterStmt == nil && len(p.counterLengths) != 0 {
		panic("internal error: seen functions with regonly/testmain")
	}

	// Emit package name.
	fmt.Fprintf(w, "\npackage %s\n\n", pkgconfig.PkgName)

	// Emit package ID var.
	fmt.Fprintf(w, "\nvar %sP uint32\n", *varVar)

	// Emit all of the counter variables.
	for k := range p.counterLengths {
		cvn := mkCounterVarName(k)
		fmt.Fprintf(w, "var %s [%d]uint32\n", cvn, p.counterLengths[k])
	}

	// Emit encoded meta-data.
	var sws slicewriter.WriteSeeker
	digest, err := p.mdb.Emit(&sws)
	if err != nil {
		log.Fatalf("encoding meta-data: %v", err)
	}
	p.mdb = nil
	fmt.Fprintf(w, "var %s = [...]byte{\n", mkMetaVar())
	payload := sws.BytesWritten()
	for k, b := range payload {
		fmt.Fprintf(w, " 0x%x,", b)
		if k != 0 && k%8 == 0 {
			fmt.Fprintf(w, "\n")
		}
	}
	fmt.Fprintf(w, "}\n")

	fixcfg := covcmd.CoverFixupConfig{
		Strategy:           "normal",
		MetaVar:            mkMetaVar(),
		MetaLen:            len(payload),
		MetaHash:           fmt.Sprintf("%x", digest),
		PkgIdVar:           mkPackageIdVar(),
		CounterPrefix:      *varVar,
		CounterGranularity: pkgconfig.Granularity,
		CounterMode:        *mode,
	}
	fixdata, err := json.Marshal(fixcfg)
	if err != nil {
		log.Fatalf("marshal fixupcfg: %v", err)
	}
	if err := os.WriteFile(pkgconfig.OutConfig, fixdata, 0666); err != nil {
		log.Fatalf("error writing %s: %v", pkgconfig.OutConfig, err)
	}
}

// atomicOnAtomic returns true if we're instrumenting
// the sync/atomic package AND using atomic mode.
func atomicOnAtomic() bool {
	return *mode == "atomic" && pkgconfig.PkgPath == "sync/atomic"
}

// atomicPackagePrefix returns the import path prefix used to refer to
// our special import of sync/atomic; this is either set to the
// constant atomicPackageName plus a dot or the empty string if we're
// instrumenting the sync/atomic package itself.
func atomicPackagePrefix() string {
	if atomicOnAtomic() {
		return ""
	}
	return atomicPackageName + "."
}

func (p *Package) emitMetaFile(outpath string) {
	// Open output file.
	of, err := os.OpenFile(outpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		log.Fatalf("opening covmeta %s: %v", outpath, err)
	}

	if len(p.counterLengths) == 0 {
		// This corresponds to the case where we have no functions
		// in the package to instrument. Leave the file empty file if
		// this happens.
		if err = of.Close(); err != nil {
			log.Fatalf("closing meta-data file: %v", err)
		}
		return
	}

	// Encode meta-data.
	var sws slicewriter.WriteSeeker
	digest, err := p.mdb.Emit(&sws)
	if err != nil {
		log.Fatalf("encoding meta-data: %v", err)
	}
	payload := sws.BytesWritten()
	blobs := [][]byte{payload}

	// Write meta-data file directly.
	mfw := encodemeta.NewCoverageMetaFileWriter(outpath, of)
	err = mfw.Write(digest, blobs, cmode, cgran)
	if err != nil {
		log.Fatalf("writing meta-data file: %v", err)
	}
	if err = of.Close(); err != nil {
		log.Fatalf("closing meta-data file: %v", err)
	}
}
