// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cover is a program that is used by 'go test -cover' to rewrite the source code
// with annotations to track which parts of each function are executed.
// It operates on one Go source file at a time, computing approximate
// basic block information by studying the source. It is thus more portable
// than binary-rewriting coverage tools, but also a little less capable.
// For instance, it does not probe inside && and || expressions, and can
// be mildly confused by single statements with multiple function literals.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strconv"
)

var (
	mode     = flag.String("mode", "set", "coverage mode: set, count, atomic")
	countVar = flag.String("count", "__count", "name of coverage count array variable")
	posVar   = flag.String("pos", "__pos", "name of coverage count position variable")
)

var counterStmt func(*File, ast.Expr) ast.Stmt

const (
	atomicPackagePath = "sync/atomic"
	atomicPackageName = "_cover_atomic_"
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: %s [options] file\n", os.Args[0])
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	flag.Usage = usage
	flag.Parse()
	switch *mode {
	case "set":
		counterStmt = setCounterStmt
	case "count":
		counterStmt = incCounterStmt
	case "atomic":
		counterStmt = atomicCounterStmt
	default:
		flag.Usage()
		os.Exit(2)
	}
	if flag.NArg() != 1 {
		flag.Usage()
	}
	cover(flag.Arg(0))
}

// Block represents the information about a basic block to be recorded in the analysis.
// Note: Our definition of basic block is based on control structures; we don't break
// apart && and ||. We could but it doesn't seem important enough to bother.
type Block struct {
	startByte token.Pos
	endByte   token.Pos
}

// File is a wrapper for the state of a file used in the parser.
// The basic parse tree walker is a method of this type.
type File struct {
	fset      *token.FileSet
	name      string // Name of file.
	astFile   *ast.File
	blocks    []Block
	atomicPkg string // Package name for "sync/atomic" in this file.
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
					clause.Body = f.addCounters(clause.Pos(), clause.End(), clause.Body)
				}
				return f
			case *ast.CommClause: // select
				for _, n := range n.List {
					clause := n.(*ast.CommClause)
					clause.Body = f.addCounters(clause.Pos(), clause.End(), clause.Body)
				}
				return f
			}
		}
		n.List = f.addCounters(n.Pos(), n.End(), n.List)
	}
	return f
}

// unquote returns the unquoted string.
func unquote(s string) string {
	t, err := strconv.Unquote(s)
	if err != nil {
		log.Fatal("cover: improperly quoted string %q\n", s)
	}
	return t
}

// addImport adds an import for the specified path, if one does not already exist, and returns
// the local package name.
func (f *File) addImport(path string) string {
	// Does the package already import it?
	for _, s := range f.astFile.Imports {
		if unquote(s.Path.Value) == path {
			return s.Name.Name
		}
	}
	newImport := &ast.ImportSpec{
		Name: ast.NewIdent(atomicPackageName),
		Path: &ast.BasicLit{
			Kind:  token.STRING,
			Value: fmt.Sprintf("%q", path),
		},
	}
	impDecl := &ast.GenDecl{
		Tok: token.IMPORT,
		Specs: []ast.Spec{
			newImport,
		},
	}
	// Make the new import the first Decl in the file.
	astFile := f.astFile
	astFile.Decls = append(astFile.Decls, nil)
	copy(astFile.Decls[1:], astFile.Decls[0:])
	astFile.Decls[0] = impDecl
	astFile.Imports = append(astFile.Imports, newImport)

	// Now refer to the package, just in case it ends up unused.
	// That is, append to the end of the file the declaration
	//	var _ = _cover_atomic_.AddUint32
	reference := &ast.GenDecl{
		Tok: token.VAR,
		Specs: []ast.Spec{
			&ast.ValueSpec{
				Names: []*ast.Ident{
					ast.NewIdent("_"),
				},
				Values: []ast.Expr{
					&ast.SelectorExpr{
						X:   ast.NewIdent(atomicPackageName),
						Sel: ast.NewIdent("AddUint32"),
					},
				},
			},
		},
	}
	astFile.Decls = append(astFile.Decls, reference)
	return atomicPackageName
}

func cover(name string) {
	var files []*File
	var astFiles []*ast.File
	fs := token.NewFileSet()
	f, err := os.Open(name)
	if err != nil {
		log.Fatalf("cover: %s: %s", name, err)
	}
	defer f.Close()
	data, err := ioutil.ReadAll(f)
	if err != nil {
		log.Fatalf("cover: %s: %s", name, err)
	}
	parsedFile, err := parser.ParseFile(fs, name, bytes.NewReader(data), 0)
	if err != nil {
		log.Fatalf("cover: %s: %s", name, err)
	}
	thisFile := &File{
		fset:    fs,
		name:    name,
		astFile: parsedFile,
	}
	files = append(files, thisFile)
	astFiles = append(astFiles, parsedFile)
	for _, file := range files {
		if *mode == "atomic" {
			file.atomicPkg = file.addImport(atomicPackagePath)
		}
		ast.Walk(file, file.astFile)
		file.print(os.Stdout)
		// After printing the source tree, add some declarations for the counters etc.
		// We could do this by adding to the tree, but it's easier just to print the text.
		file.addVariables(os.Stdout)
	}
}

func (f *File) print(w io.Writer) {
	printer.Fprint(w, f.fset, f.astFile)
}

// intLiteral returns an ast.BasicLit representing the integer value.
func (f *File) intLiteral(i int) *ast.BasicLit {
	node := &ast.BasicLit{
		Kind:  token.INT,
		Value: fmt.Sprint(i),
	}
	return node
}

// index returns an ast.BasicLit representing the number of counters present.
func (f *File) index() *ast.BasicLit {
	return f.intLiteral(len(f.blocks))
}

// setCounterStmt returns the expression: __count[23] = 1.
func setCounterStmt(f *File, counter ast.Expr) ast.Stmt {
	return &ast.AssignStmt{
		Lhs: []ast.Expr{counter},
		Tok: token.ASSIGN,
		Rhs: []ast.Expr{f.intLiteral(1)},
	}
}

// incCounterStmt returns the expression: __count[23]++.
func incCounterStmt(f *File, counter ast.Expr) ast.Stmt {
	return &ast.IncDecStmt{
		X:   counter,
		Tok: token.INC,
	}
}

// atomicCounterStmt returns the expression: atomic.AddUint32(&__count[23], 1)
func atomicCounterStmt(f *File, counter ast.Expr) ast.Stmt {
	return &ast.ExprStmt{
		X: &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   ast.NewIdent(f.atomicPkg),
				Sel: ast.NewIdent("AddUint32"),
			},
			Args: []ast.Expr{&ast.UnaryExpr{
				Op: token.AND,
				X:  counter,
			},
				f.intLiteral(1),
			},
		},
	}
}

// newCounter creates a new counter expression of the appropriate form.
func (f *File) newCounter(start, end token.Pos) ast.Stmt {
	counter := &ast.IndexExpr{
		X:     ast.NewIdent(*countVar),
		Index: f.index(),
	}
	stmt := counterStmt(f, counter)
	f.blocks = append(f.blocks, Block{start, end})
	return stmt
}

// addCounters takes a list of statements and adds counters to the beginning of
// each basic block at the top level of that list. For instance, given
//
//	S1
//	if cond {
//		S2
// 	}
//	S3
//
// counters will be added before S1 and before S3. The block containing S2
// will be visited in a separate call.
// TODO: Nested simple blocks get unecessary (but correct) counters
func (f *File) addCounters(pos, end token.Pos, list []ast.Stmt) []ast.Stmt {
	// Special case: make sure we add a counter to an empty block. Can't do this below
	// or we will add a counter to an empty statement list after, say, a return statement.
	if len(list) == 0 {
		return []ast.Stmt{f.newCounter(pos, end)}
	}
	// We have a block (statement list), but it may have several basic blocks due to the
	// appearance of statements that affect the flow of control.
	var newList []ast.Stmt
	for {
		// Find first statement that affects flow of control (break, continue, if, etc.).
		// It will be the last statement of this basic block.
		var last int
		end = pos
		for last = 0; last < len(list); last++ {
			end = f.statementBoundary(list[last])
			if f.endsBasicSourceBlock(list[last]) {
				last++
				break
			}
		}
		if pos != end { // Can have no source to cover if e.g. blocks abut.
			newList = append(newList, f.newCounter(pos, end))
		}
		newList = append(newList, list[0:last]...)
		list = list[last:]
		if len(list) == 0 {
			break
		}
		pos = list[0].Pos()
	}
	return newList
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
		return s.Body.Lbrace
	case *ast.ForStmt:
		return s.Body.Lbrace
	case *ast.LabeledStmt:
		return f.statementBoundary(s.Stmt)
	case *ast.RangeStmt:
		return s.Body.Lbrace
	case *ast.SwitchStmt:
		return s.Body.Lbrace
	case *ast.SelectStmt:
		return s.Body.Lbrace
	case *ast.TypeSwitchStmt:
		return s.Body.Lbrace
	}
	// If not a control flow statement, it is a declaration, expression, call, etc. and it may have a function literal.
	// If it does, that's tricky because we want to exclude the body of the function from this block.
	// Draw a line at the start of the body of the first function literal we find.
	// TODO: what if there's more than one? Probably doesn't matter much.
	var literal funcLitFinder
	ast.Walk(&literal, s)
	if literal.found() {
		return token.Pos(literal)
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
		return f.endsBasicSourceBlock(s.Stmt)
	case *ast.RangeStmt:
		return true
	case *ast.SwitchStmt:
		return true
	case *ast.SelectStmt:
		return true
	case *ast.TypeSwitchStmt:
		return true
	}
	var literal funcLitFinder
	ast.Walk(&literal, s)
	return literal.found()
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

// addVariables adds to the end of the file the declarations to set up the counter and position variables.
func (f *File) addVariables(w io.Writer) {
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
			fmt.Fprintf(os.Stderr, "\t%s:#%d,#%d %s:#%d,#%d\n", f.name, t[i-1].startByte, t[i-1].endByte, f.name, t[i].startByte, t[i].endByte)
		}
	}

	// Declare the coverage array as a package-level variable.
	// Everything else will be local to init.
	fmt.Fprintf(w, "\nvar %s [%d]uint32\n\n", *countVar, len(f.blocks))

	// Declare the position array as a package-level variable.
	fmt.Fprintf(w, "var %s = [3*%d]uint32{\n", *posVar, len(f.blocks))

	// Here's a nice long list of positions. Each position is encoded as follows to reduce size:
	// - 32-bit starting line number
	// - 32-bit ending line number
	// - (16 bit ending column number << 16) | (16-bit starting column number).
	for _, block := range f.blocks {
		start := f.fset.Position(block.startByte)
		end := f.fset.Position(block.endByte)
		fmt.Fprintf(w, "\t%d, %d, %#x,\n", start.Line, end.Line, (end.Column&0xFFFF)<<16|(start.Column&0xFFFF))
	}

	// Close the declaration.
	fmt.Fprintf(w, "}\n")
}
