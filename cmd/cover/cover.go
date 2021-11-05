// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	"path/filepath"
	"sort"
	"strconv"
	"strings"
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
(what go test -cover does):
	go tool cover -mode=set -var=CoverageVariableName program.go
`

func usage() {
	fmt.Fprint(os.Stderr, usageMessage)
	fmt.Fprintln(os.Stderr, "\nFlags:")
	flag.PrintDefaults()
	fmt.Fprintln(os.Stderr, "\n  Only one of -html, -func, or -mode may be set.")
	os.Exit(2)
}

var (
	mode    = flag.String("mode", "", "coverage mode: set, count, atomic")
	varVar  = flag.String("var", "GoCover", "name of coverage variable to generate")
	output  = flag.String("o", "", "file for output; default: stdout")
	htmlOut = flag.String("html", "", "generate HTML representation of coverage profile")
	funcOut = flag.String("func", "", "output coverage profile information for each function")
)

var profile string // The profile to read; the value of -html or -func

var counterStmt func(*File, ast.Expr) ast.Stmt

const (
	atomicPackagePath = "sync/atomic"
	atomicPackageName = "_cover_atomic_"
)

func main() {
	flag.Usage = usage
	flag.Parse()

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
		annotate(flag.Arg(0))
		return
	}

	// Output HTML or function coverage information.
	if *htmlOut != "" {
		err = htmlOutput(profile, *output)
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

	// Must either display a profile or rewrite Go source.
	if (profile == "") == (*mode == "") {
		return fmt.Errorf("too many options")
	}

	if *mode != "" {
		switch *mode {
		case "set":
			counterStmt = setCounterStmt
		case "count":
			counterStmt = incCounterStmt
		case "atomic":
			counterStmt = atomicCounterStmt
		default:
			return fmt.Errorf("unknown -mode %v", *mode)
		}

		if flag.NArg() == 0 {
			return fmt.Errorf("missing source file")
		} else if flag.NArg() == 1 {
			return nil
		}
	} else if flag.NArg() == 0 {
		return nil
	}
	return fmt.Errorf("too many arguments")
}

// Block represents the information about a basic block to be recorded in the analysis.
// Note: Our definition of basic block is based on control structures; we don't break
// apart && and ||. We could but it doesn't seem important enough to bother.
type Block struct {
	startByte token.Pos
	endByte   token.Pos
	numStmt   int
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
					clause.Body = f.addCounters(clause.Pos(), clause.End(), clause.Body, false)
				}
				return f
			case *ast.CommClause: // select
				for _, n := range n.List {
					clause := n.(*ast.CommClause)
					clause.Body = f.addCounters(clause.Pos(), clause.End(), clause.Body, false)
				}
				return f
			}
		}
		n.List = f.addCounters(n.Lbrace, n.Rbrace+1, n.List, true) // +1 to step past closing brace.
	case *ast.IfStmt:
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
		switch stmt := n.Else.(type) {
		case *ast.IfStmt:
			block := &ast.BlockStmt{
				Lbrace: n.Body.End(), // Start at end of the "if" block so the covered part looks like it starts at the "else".
				List:   []ast.Stmt{stmt},
				Rbrace: stmt.End(),
			}
			n.Else = block
		case *ast.BlockStmt:
			stmt.Lbrace = n.Body.End() // Start at end of the "if" block so the covered part looks like it starts at the "else".
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
			return nil
		}
	case *ast.TypeSwitchStmt:
		// Don't annotate an empty type switch - creates a syntax error.
		if n.Body == nil || len(n.Body.List) == 0 {
			return nil
		}
	}
	return f
}

// unquote returns the unquoted string.
func unquote(s string) string {
	t, err := strconv.Unquote(s)
	if err != nil {
		log.Fatalf("cover: improperly quoted string %q\n", s)
	}
	return t
}

// addImport adds an import for the specified path, if one does not already exist, and returns
// the local package name.
func (f *File) addImport(path string) string {
	// Does the package already import it?
	for _, s := range f.astFile.Imports {
		if unquote(s.Path.Value) == path {
			if s.Name != nil {
				return s.Name.Name
			}
			return filepath.Base(path)
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

var slashslash = []byte("//")

// initialComments returns the prefix of content containing only
// whitespace and line comments.  Any +build directives must appear
// within this region.  This approach is more reliable than using
// go/printer to print a modified AST containing comments.
//
func initialComments(content []byte) []byte {
	// Derived from go/build.Context.shouldBuild.
	end := 0
	p := content
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 { // Blank line.
			end = len(content) - len(p)
			continue
		}
		if !bytes.HasPrefix(line, slashslash) { // Not comment line.
			break
		}
	}
	return content[:end]
}

func annotate(name string) {
	fset := token.NewFileSet()
	content, err := ioutil.ReadFile(name)
	if err != nil {
		log.Fatalf("cover: %s: %s", name, err)
	}
	parsedFile, err := parser.ParseFile(fset, name, content, parser.ParseComments)
	if err != nil {
		log.Fatalf("cover: %s: %s", name, err)
	}
	parsedFile.Comments = trimComments(parsedFile, fset)

	file := &File{
		fset:    fset,
		name:    name,
		astFile: parsedFile,
	}
	if *mode == "atomic" {
		file.atomicPkg = file.addImport(atomicPackagePath)
	}
	ast.Walk(file, file.astFile)
	fd := os.Stdout
	if *output != "" {
		var err error
		fd, err = os.Create(*output)
		if err != nil {
			log.Fatalf("cover: %s", err)
		}
	}
	fd.Write(initialComments(content)) // Retain '// +build' directives.
	file.print(fd)
	// After printing the source tree, add some declarations for the counters etc.
	// We could do this by adding to the tree, but it's easier just to print the text.
	file.addVariables(fd)
}

// trimComments drops all but the //go: comments, some of which are semantically important.
// We drop all others because they can appear in places that cause our counters
// to appear in syntactically incorrect places. //go: appears at the beginning of
// the line and is syntactically safe.
func trimComments(file *ast.File, fset *token.FileSet) []*ast.CommentGroup {
	var comments []*ast.CommentGroup
	for _, group := range file.Comments {
		var list []*ast.Comment
		for _, comment := range group.List {
			if strings.HasPrefix(comment.Text, "//go:") && fset.Position(comment.Slash).Column == 1 {
				list = append(list, comment)
			}
		}
		if list != nil {
			comments = append(comments, &ast.CommentGroup{List: list})
		}
	}
	return comments
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
func (f *File) newCounter(start, end token.Pos, numStmt int) ast.Stmt {
	counter := &ast.IndexExpr{
		X: &ast.SelectorExpr{
			X:   ast.NewIdent(*varVar),
			Sel: ast.NewIdent("Count"),
		},
		Index: f.index(),
	}
	stmt := counterStmt(f, counter)
	f.blocks = append(f.blocks, Block{start, end, numStmt})
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
// TODO: Nested simple blocks get unnecessary (but correct) counters
func (f *File) addCounters(pos, blockEnd token.Pos, list []ast.Stmt, extendToClosingBrace bool) []ast.Stmt {
	// Special case: make sure we add a counter to an empty block. Can't do this below
	// or we will add a counter to an empty statement list after, say, a return statement.
	if len(list) == 0 {
		return []ast.Stmt{f.newCounter(pos, blockEnd, 0)}
	}
	// We have a block (statement list), but it may have several basic blocks due to the
	// appearance of statements that affect the flow of control.
	var newList []ast.Stmt
	for {
		// Find first statement that affects flow of control (break, continue, if, etc.).
		// It will be the last statement of this basic block.
		var last int
		end := blockEnd
		for last = 0; last < len(list); last++ {
			end = f.statementBoundary(list[last])
			if f.endsBasicSourceBlock(list[last]) {
				extendToClosingBrace = false // Block is broken up now.
				last++
				break
			}
		}
		if extendToClosingBrace {
			end = blockEnd
		}
		if pos != end { // Can have no source to cover if e.g. blocks abut.
			newList = append(newList, f.newCounter(pos, end, last))
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
		return f.endsBasicSourceBlock(s.Stmt)
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
