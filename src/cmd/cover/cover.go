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
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"sort"

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
(what go test -cover does):
	go tool cover -mode=set -var=CoverageVariableName program.go
`

func usage() {
	fmt.Fprintln(os.Stderr, usageMessage)
	fmt.Fprintln(os.Stderr, "Flags:")
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

var counterStmt func(*File, string) string

const (
	atomicPackagePath = "sync/atomic"
	atomicPackageName = "_cover_atomic_"
)

func main() {
	objabi.AddVersionFlag()
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

	if *varVar != "" && !token.IsIdentifier(*varVar) {
		return fmt.Errorf("-var: %q is not a valid identifier", *varVar)
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
	fset    *token.FileSet
	name    string // Name of file.
	astFile *ast.File
	blocks  []Block
	content []byte
	edit    *edit.Buffer
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
		if n.Name.Name == "_" {
			return nil
		}
	}
	return f
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

	file := &File{
		fset:    fset,
		name:    name,
		content: content,
		edit:    edit.NewBuffer(content),
		astFile: parsedFile,
	}
	if *mode == "atomic" {
		// Add import of sync/atomic immediately after package clause.
		// We do this even if there is an existing import, because the
		// existing import may be shadowed at any given place we want
		// to refer to it, and our name (_cover_atomic_) is less likely to
		// be shadowed.
		file.edit.Insert(file.offset(file.astFile.Name.End()),
			fmt.Sprintf("; import %s %q", atomicPackageName, atomicPackagePath))
	}

	ast.Walk(file, file.astFile)
	newContent := file.edit.Bytes()

	fd := os.Stdout
	if *output != "" {
		var err error
		fd, err = os.Create(*output)
		if err != nil {
			log.Fatalf("cover: %s", err)
		}
	}

	fmt.Fprintf(fd, "//line %s:1\n", name)
	fd.Write(newContent)

	// After printing the source tree, add some declarations for the counters etc.
	// We could do this by adding to the tree, but it's easier just to print the text.
	file.addVariables(fd)
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
	return fmt.Sprintf("%s.AddUint32(&%s, 1)", atomicPackageName, counter)
}

// newCounter creates a new counter expression of the appropriate form.
func (f *File) newCounter(start, end token.Pos, numStmt int) string {
	stmt := counterStmt(f, fmt.Sprintf("%s.Count[%d]", *varVar, len(f.blocks)))
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

	// Emit a reference to the atomic package to avoid
	// import and not used error when there's no code in a file.
	if *mode == "atomic" {
		fmt.Fprintf(w, "var _ = %s.LoadUint32\n", atomicPackageName)
	}
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
