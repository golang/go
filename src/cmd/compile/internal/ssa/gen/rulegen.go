// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gen

// This program generates Go code that applies rewrite rules to a Value.
// The generated code implements a function of type func (v *Value) bool
// which reports whether if did something.
// Ideas stolen from Swift: http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-2000-2.html

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

// rule syntax:
//  sexpr [&& extra conditions] -> [@block] sexpr
//
// sexpr are s-expressions (lisp-like parenthesized groupings)
// sexpr ::= [variable:](opcode sexpr*)
//         | variable
//         | <type>
//         | [auxint]
//         | {aux}
//
// aux      ::= variable | {code}
// type     ::= variable | {code}
// variable ::= some token
// opcode   ::= one of the opcodes from the *Ops.go files

// extra conditions is just a chunk of Go that evaluates to a boolean. It may use
// variables declared in the matching sexpr. The variable "v" is predefined to be
// the value matched by the entire rule.

// If multiple rules match, the first one in file order is selected.

var (
	genLog  = flag.Bool("log", false, "generate code that logs; for debugging only")
	addLine = flag.Bool("line", false, "add line number comment to generated rules; for debugging only")
)

type Rule struct {
	rule string
	loc  string // file name & line number
}

func (r Rule) String() string {
	return fmt.Sprintf("rule %q at %s", r.rule, r.loc)
}

func normalizeSpaces(s string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
}

// parse returns the matching part of the rule, additional conditions, and the result.
func (r Rule) parse() (match, cond, result string) {
	s := strings.Split(r.rule, "->")
	if len(s) != 2 {
		log.Fatalf("no arrow in %s", r)
	}
	match = normalizeSpaces(s[0])
	result = normalizeSpaces(s[1])
	cond = ""
	if i := strings.Index(match, "&&"); i >= 0 {
		cond = normalizeSpaces(match[i+2:])
		match = normalizeSpaces(match[:i])
	}
	return match, cond, result
}

func genRules(arch arch)          { genRulesSuffix(arch, "") }
func genSplitLoadRules(arch arch) { genRulesSuffix(arch, "splitload") }

func genRulesSuffix(arch arch, suff string) {
	// Open input file.
	text, err := os.Open(arch.name + suff + ".rules")
	if err != nil {
		if suff == "" {
			// All architectures must have a plain rules file.
			log.Fatalf("can't read rule file: %v", err)
		}
		// Some architectures have bonus rules files that others don't share. That's fine.
		return
	}

	// oprules contains a list of rules for each block and opcode
	blockrules := map[string][]Rule{}
	oprules := map[string][]Rule{}

	// read rule file
	scanner := bufio.NewScanner(text)
	rule := ""
	var lineno int
	var ruleLineno int // line number of "->"
	for scanner.Scan() {
		lineno++
		line := scanner.Text()
		if i := strings.Index(line, "//"); i >= 0 {
			// Remove comments. Note that this isn't string safe, so
			// it will truncate lines with // inside strings. Oh well.
			line = line[:i]
		}
		rule += " " + line
		rule = strings.TrimSpace(rule)
		if rule == "" {
			continue
		}
		if !strings.Contains(rule, "->") {
			continue
		}
		if ruleLineno == 0 {
			ruleLineno = lineno
		}
		if strings.HasSuffix(rule, "->") {
			continue
		}
		if unbalanced(rule) {
			continue
		}

		loc := fmt.Sprintf("%s%s.rules:%d", arch.name, suff, ruleLineno)
		for _, rule2 := range expandOr(rule) {
			r := Rule{rule: rule2, loc: loc}
			if rawop := strings.Split(rule2, " ")[0][1:]; isBlock(rawop, arch) {
				blockrules[rawop] = append(blockrules[rawop], r)
				continue
			}
			// Do fancier value op matching.
			match, _, _ := r.parse()
			op, oparch, _, _, _, _ := parseValue(match, arch, loc)
			opname := fmt.Sprintf("Op%s%s", oparch, op.name)
			oprules[opname] = append(oprules[opname], r)
		}
		rule = ""
		ruleLineno = 0
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("scanner failed: %v\n", err)
	}
	if unbalanced(rule) {
		log.Fatalf("%s.rules:%d: unbalanced rule: %v\n", arch.name, lineno, rule)
	}

	// Order all the ops.
	var ops []string
	for op := range oprules {
		ops = append(ops, op)
	}
	sort.Strings(ops)

	genFile := &File{arch: arch, suffix: suff}
	// Main rewrite routine is a switch on v.Op.
	fn := &Func{kind: "Value", arglen: -1}

	sw := &Switch{expr: exprf("v.Op")}
	for _, op := range ops {
		eop, ok := parseEllipsisRules(oprules[op], arch)
		if ok {
			swc := &Case{expr: exprf(op)}
			swc.add(stmtf("v.Op = %s", eop))
			swc.add(stmtf("return true"))
			sw.add(swc)
			continue
		}

		swc := &Case{expr: exprf(op)}
		swc.add(stmtf("return rewriteValue%s%s_%s(v)", arch.name, suff, op))
		sw.add(swc)
	}
	fn.add(sw)
	fn.add(stmtf("return false"))
	genFile.add(fn)

	// Generate a routine per op. Note that we don't make one giant routine
	// because it is too big for some compilers.
	for _, op := range ops {
		rules := oprules[op]
		_, ok := parseEllipsisRules(oprules[op], arch)
		if ok {
			continue
		}

		// rr is kept between iterations, so that each rule can check
		// that the previous rule wasn't unconditional.
		var rr *RuleRewrite
		fn := &Func{
			kind:   "Value",
			suffix: fmt.Sprintf("_%s", op),
			arglen: opByName(arch, op).argLength,
		}
		fn.add(declf("b", "v.Block"))
		fn.add(declf("config", "b.Func.Config"))
		fn.add(declf("fe", "b.Func.fe"))
		fn.add(declf("typ", "&b.Func.Config.Types"))
		for _, rule := range rules {
			if rr != nil && !rr.canFail {
				log.Fatalf("unconditional rule %s is followed by other rules", rr.match)
			}
			rr = &RuleRewrite{loc: rule.loc}
			rr.match, rr.cond, rr.result = rule.parse()
			pos, _ := genMatch(rr, arch, rr.match, fn.arglen >= 0)
			if pos == "" {
				pos = "v.Pos"
			}
			if rr.cond != "" {
				rr.add(breakf("!(%s)", rr.cond))
			}
			genResult(rr, arch, rr.result, pos)
			if *genLog {
				rr.add(stmtf("logRule(%q)", rule.loc))
			}
			fn.add(rr)
		}
		if rr.canFail {
			fn.add(stmtf("return false"))
		}
		genFile.add(fn)
	}

	// Generate block rewrite function. There are only a few block types
	// so we can make this one function with a switch.
	fn = &Func{kind: "Block"}
	fn.add(declf("config", "b.Func.Config"))
	fn.add(declf("typ", "&b.Func.Config.Types"))

	sw = &Switch{expr: exprf("b.Kind")}
	ops = ops[:0]
	for op := range blockrules {
		ops = append(ops, op)
	}
	sort.Strings(ops)
	for _, op := range ops {
		name, data := getBlockInfo(op, arch)
		swc := &Case{expr: exprf("%s", name)}
		for _, rule := range blockrules[op] {
			swc.add(genBlockRewrite(rule, arch, data))
		}
		sw.add(swc)
	}
	fn.add(sw)
	fn.add(stmtf("return false"))
	genFile.add(fn)

	// Remove unused imports and variables.
	buf := new(bytes.Buffer)
	fprint(buf, genFile)
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", buf, parser.ParseComments)
	if err != nil {
		filename := fmt.Sprintf("%s_broken.go", arch.name)
		if err := ioutil.WriteFile(filename, buf.Bytes(), 0644); err != nil {
			log.Printf("failed to dump broken code to %s: %v", filename, err)
		} else {
			log.Printf("dumped broken code to %s", filename)
		}
		log.Fatalf("failed to parse generated code for arch %s: %v", arch.name, err)
	}
	tfile := fset.File(file.Pos())

	// First, use unusedInspector to find the unused declarations by their
	// start position.
	u := unusedInspector{unused: make(map[token.Pos]bool)}
	u.node(file)

	// Then, delete said nodes via astutil.Apply.
	pre := func(c *astutil.Cursor) bool {
		node := c.Node()
		if node == nil {
			return true
		}
		if u.unused[node.Pos()] {
			c.Delete()
			// Unused imports and declarations use exactly
			// one line. Prevent leaving an empty line.
			tfile.MergeLine(tfile.Position(node.Pos()).Line)
			return false
		}
		return true
	}
	post := func(c *astutil.Cursor) bool {
		switch node := c.Node().(type) {
		case *ast.GenDecl:
			if len(node.Specs) == 0 {
				// Don't leave a broken or empty GenDecl behind,
				// such as "import ()".
				c.Delete()
			}
		}
		return true
	}
	file = astutil.Apply(file, pre, post).(*ast.File)

	// Write the well-formatted source to file
	f, err := os.Create("../rewrite" + arch.name + suff + ".go")
	if err != nil {
		log.Fatalf("can't write output: %v", err)
	}
	defer f.Close()
	// gofmt result; use a buffered writer, as otherwise go/format spends
	// far too much time in syscalls.
	bw := bufio.NewWriter(f)
	if err := format.Node(bw, fset, file); err != nil {
		log.Fatalf("can't format output: %v", err)
	}
	if err := bw.Flush(); err != nil {
		log.Fatalf("can't write output: %v", err)
	}
	if err := f.Close(); err != nil {
		log.Fatalf("can't write output: %v", err)
	}
}

// unusedInspector can be used to detect unused variables and imports in an
// ast.Node via its node method. The result is available in the "unused" map.
//
// note that unusedInspector is lazy and best-effort; it only supports the node
// types and patterns used by the rulegen program.
type unusedInspector struct {
	// scope is the current scope, which can never be nil when a declaration
	// is encountered. That is, the unusedInspector.node entrypoint should
	// generally be an entire file or block.
	scope *scope

	// unused is the resulting set of unused declared names, indexed by the
	// starting position of the node that declared the name.
	unused map[token.Pos]bool

	// defining is the object currently being defined; this is useful so
	// that if "foo := bar" is unused and removed, we can then detect if
	// "bar" becomes unused as well.
	defining *object
}

// scoped opens a new scope when called, and returns a function which closes
// that same scope. When a scope is closed, unused variables are recorded.
func (u *unusedInspector) scoped() func() {
	outer := u.scope
	u.scope = &scope{outer: outer, objects: map[string]*object{}}
	return func() {
		for anyUnused := true; anyUnused; {
			anyUnused = false
			for _, obj := range u.scope.objects {
				if obj.numUses > 0 {
					continue
				}
				u.unused[obj.pos] = true
				for _, used := range obj.used {
					if used.numUses--; used.numUses == 0 {
						anyUnused = true
					}
				}
				// We've decremented numUses for each of the
				// objects in used. Zero this slice too, to keep
				// everything consistent.
				obj.used = nil
			}
		}
		u.scope = outer
	}
}

func (u *unusedInspector) exprs(list []ast.Expr) {
	for _, x := range list {
		u.node(x)
	}
}

func (u *unusedInspector) node(node ast.Node) {
	switch node := node.(type) {
	case *ast.File:
		defer u.scoped()()
		for _, decl := range node.Decls {
			u.node(decl)
		}
	case *ast.GenDecl:
		for _, spec := range node.Specs {
			u.node(spec)
		}
	case *ast.ImportSpec:
		impPath, _ := strconv.Unquote(node.Path.Value)
		name := path.Base(impPath)
		u.scope.objects[name] = &object{
			name: name,
			pos:  node.Pos(),
		}
	case *ast.FuncDecl:
		u.node(node.Type)
		if node.Body != nil {
			u.node(node.Body)
		}
	case *ast.FuncType:
		if node.Params != nil {
			u.node(node.Params)
		}
		if node.Results != nil {
			u.node(node.Results)
		}
	case *ast.FieldList:
		for _, field := range node.List {
			u.node(field)
		}
	case *ast.Field:
		u.node(node.Type)

	// statements

	case *ast.BlockStmt:
		defer u.scoped()()
		for _, stmt := range node.List {
			u.node(stmt)
		}
	case *ast.IfStmt:
		if node.Init != nil {
			u.node(node.Init)
		}
		u.node(node.Cond)
		u.node(node.Body)
		if node.Else != nil {
			u.node(node.Else)
		}
	case *ast.ForStmt:
		if node.Init != nil {
			u.node(node.Init)
		}
		if node.Cond != nil {
			u.node(node.Cond)
		}
		if node.Post != nil {
			u.node(node.Post)
		}
		u.node(node.Body)
	case *ast.SwitchStmt:
		if node.Init != nil {
			u.node(node.Init)
		}
		if node.Tag != nil {
			u.node(node.Tag)
		}
		u.node(node.Body)
	case *ast.CaseClause:
		u.exprs(node.List)
		defer u.scoped()()
		for _, stmt := range node.Body {
			u.node(stmt)
		}
	case *ast.BranchStmt:
	case *ast.ExprStmt:
		u.node(node.X)
	case *ast.AssignStmt:
		if node.Tok != token.DEFINE {
			u.exprs(node.Rhs)
			u.exprs(node.Lhs)
			break
		}
		if len(node.Lhs) != 1 {
			panic("no support for := with multiple names")
		}

		name := node.Lhs[0].(*ast.Ident)
		obj := &object{
			name: name.Name,
			pos:  name.NamePos,
		}

		old := u.defining
		u.defining = obj
		u.exprs(node.Rhs)
		u.defining = old

		u.scope.objects[name.Name] = obj
	case *ast.ReturnStmt:
		u.exprs(node.Results)
	case *ast.IncDecStmt:
		u.node(node.X)

	// expressions

	case *ast.CallExpr:
		u.node(node.Fun)
		u.exprs(node.Args)
	case *ast.SelectorExpr:
		u.node(node.X)
	case *ast.UnaryExpr:
		u.node(node.X)
	case *ast.BinaryExpr:
		u.node(node.X)
		u.node(node.Y)
	case *ast.StarExpr:
		u.node(node.X)
	case *ast.ParenExpr:
		u.node(node.X)
	case *ast.IndexExpr:
		u.node(node.X)
		u.node(node.Index)
	case *ast.TypeAssertExpr:
		u.node(node.X)
		u.node(node.Type)
	case *ast.Ident:
		if obj := u.scope.Lookup(node.Name); obj != nil {
			obj.numUses++
			if u.defining != nil {
				u.defining.used = append(u.defining.used, obj)
			}
		}
	case *ast.BasicLit:
	default:
		panic(fmt.Sprintf("unhandled node: %T", node))
	}
}

// scope keeps track of a certain scope and its declared names, as well as the
// outer (parent) scope.
type scope struct {
	outer   *scope             // can be nil, if this is the top-level scope
	objects map[string]*object // indexed by each declared name
}

func (s *scope) Lookup(name string) *object {
	if obj := s.objects[name]; obj != nil {
		return obj
	}
	if s.outer == nil {
		return nil
	}
	return s.outer.Lookup(name)
}

// object keeps track of a declared name, such as a variable or import.
type object struct {
	name string
	pos  token.Pos // start position of the node declaring the object

	numUses int       // number of times this object is used
	used    []*object // objects that its declaration makes use of
}

func fprint(w io.Writer, n Node) {
	switch n := n.(type) {
	case *File:
		file := n
		seenRewrite := make(map[[3]string]string)
		fmt.Fprintf(w, "// Code generated from gen/%s%s.rules; DO NOT EDIT.\n", n.arch.name, n.suffix)
		fmt.Fprintf(w, "// generated with: cd gen; go run *.go\n")
		fmt.Fprintf(w, "\npackage ssa\n")
		for _, path := range append([]string{
			"fmt",
			"math",
			"cmd/internal/obj",
			"cmd/internal/objabi",
			"cmd/compile/internal/types",
		}, n.arch.imports...) {
			fmt.Fprintf(w, "import %q\n", path)
		}
		for _, f := range n.list {
			f := f.(*Func)
			fmt.Fprintf(w, "func rewrite%s%s%s%s(", f.kind, n.arch.name, n.suffix, f.suffix)
			fmt.Fprintf(w, "%c *%s) bool {\n", strings.ToLower(f.kind)[0], f.kind)
			if f.kind == "Value" && f.arglen > 0 {
				for i := f.arglen - 1; i >= 0; i-- {
					fmt.Fprintf(w, "v_%d := v.Args[%d]\n", i, i)
				}
			}
			for _, n := range f.list {
				fprint(w, n)

				if rr, ok := n.(*RuleRewrite); ok {
					k := [3]string{
						normalizeMatch(rr.match, file.arch),
						normalizeWhitespace(rr.cond),
						normalizeWhitespace(rr.result),
					}
					if prev, ok := seenRewrite[k]; ok {
						log.Fatalf("duplicate rule %s, previously seen at %s\n", rr.loc, prev)
					} else {
						seenRewrite[k] = rr.loc
					}
				}
			}
			fmt.Fprintf(w, "}\n")
		}
	case *Switch:
		fmt.Fprintf(w, "switch ")
		fprint(w, n.expr)
		fmt.Fprintf(w, " {\n")
		for _, n := range n.list {
			fprint(w, n)
		}
		fmt.Fprintf(w, "}\n")
	case *Case:
		fmt.Fprintf(w, "case ")
		fprint(w, n.expr)
		fmt.Fprintf(w, ":\n")
		for _, n := range n.list {
			fprint(w, n)
		}
	case *RuleRewrite:
		if *addLine {
			fmt.Fprintf(w, "// %s\n", n.loc)
		}
		fmt.Fprintf(w, "// match: %s\n", n.match)
		if n.cond != "" {
			fmt.Fprintf(w, "// cond: %s\n", n.cond)
		}
		fmt.Fprintf(w, "// result: %s\n", n.result)
		fmt.Fprintf(w, "for %s {\n", n.check)
		nCommutative := 0
		for _, n := range n.list {
			if b, ok := n.(*CondBreak); ok {
				b.insideCommuteLoop = nCommutative > 0
			}
			fprint(w, n)
			if loop, ok := n.(StartCommuteLoop); ok {
				if nCommutative != loop.depth {
					panic("mismatch commute loop depth")
				}
				nCommutative++
			}
		}
		fmt.Fprintf(w, "return true\n")
		for i := 0; i < nCommutative; i++ {
			fmt.Fprintln(w, "}")
		}
		if n.commuteDepth > 0 && n.canFail {
			fmt.Fprint(w, "break\n")
		}
		fmt.Fprintf(w, "}\n")
	case *Declare:
		fmt.Fprintf(w, "%s := ", n.name)
		fprint(w, n.value)
		fmt.Fprintln(w)
	case *CondBreak:
		fmt.Fprintf(w, "if ")
		fprint(w, n.expr)
		fmt.Fprintf(w, " {\n")
		if n.insideCommuteLoop {
			fmt.Fprintf(w, "continue")
		} else {
			fmt.Fprintf(w, "break")
		}
		fmt.Fprintf(w, "\n}\n")
	case ast.Node:
		printConfig.Fprint(w, emptyFset, n)
		if _, ok := n.(ast.Stmt); ok {
			fmt.Fprintln(w)
		}
	case StartCommuteLoop:
		fmt.Fprintf(w, "for _i%[1]d := 0; _i%[1]d <= 1; _i%[1]d, %[2]s_0, %[2]s_1 = _i%[1]d + 1, %[2]s_1, %[2]s_0 {\n", n.depth, n.v)
	default:
		log.Fatalf("cannot print %T", n)
	}
}

var printConfig = printer.Config{
	Mode: printer.RawFormat, // we use go/format later, so skip work here
}

var emptyFset = token.NewFileSet()

// Node can be a Statement or an ast.Expr.
type Node interface{}

// Statement can be one of our high-level statement struct types, or an
// ast.Stmt under some limited circumstances.
type Statement interface{}

// bodyBase is shared by all of our statement pseudo-node types which can
// contain other statements.
type bodyBase struct {
	list    []Statement
	canFail bool
}

func (w *bodyBase) add(node Statement) {
	var last Statement
	if len(w.list) > 0 {
		last = w.list[len(w.list)-1]
	}
	if node, ok := node.(*CondBreak); ok {
		w.canFail = true
		if last, ok := last.(*CondBreak); ok {
			// Add to the previous "if <cond> { break }" via a
			// logical OR, which will save verbosity.
			last.expr = &ast.BinaryExpr{
				Op: token.LOR,
				X:  last.expr,
				Y:  node.expr,
			}
			return
		}
	}

	w.list = append(w.list, node)
}

// declared reports if the body contains a Declare with the given name.
func (w *bodyBase) declared(name string) bool {
	for _, s := range w.list {
		if decl, ok := s.(*Declare); ok && decl.name == name {
			return true
		}
	}
	return false
}

// These types define some high-level statement struct types, which can be used
// as a Statement. This allows us to keep some node structs simpler, and have
// higher-level nodes such as an entire rule rewrite.
//
// Note that ast.Expr is always used as-is; we don't declare our own expression
// nodes.
type (
	File struct {
		bodyBase // []*Func
		arch     arch
		suffix   string
	}
	Func struct {
		bodyBase
		kind   string // "Value" or "Block"
		suffix string
		arglen int32 // if kind == "Value", number of args for this op
	}
	Switch struct {
		bodyBase // []*Case
		expr     ast.Expr
	}
	Case struct {
		bodyBase
		expr ast.Expr
	}
	RuleRewrite struct {
		bodyBase
		match, cond, result string // top comments
		check               string // top-level boolean expression

		alloc        int    // for unique var names
		loc          string // file name & line number of the original rule
		commuteDepth int    // used to track depth of commute loops
	}
	Declare struct {
		name  string
		value ast.Expr
	}
	CondBreak struct {
		expr              ast.Expr
		insideCommuteLoop bool
	}
	StartCommuteLoop struct {
		depth int
		v     string
	}
)

// exprf parses a Go expression generated from fmt.Sprintf, panicking if an
// error occurs.
func exprf(format string, a ...interface{}) ast.Expr {
	src := fmt.Sprintf(format, a...)
	expr, err := parser.ParseExpr(src)
	if err != nil {
		log.Fatalf("expr parse error on %q: %v", src, err)
	}
	return expr
}

// stmtf parses a Go statement generated from fmt.Sprintf. This function is only
// meant for simple statements that don't have a custom Statement node declared
// in this package, such as ast.ReturnStmt or ast.ExprStmt.
func stmtf(format string, a ...interface{}) Statement {
	src := fmt.Sprintf(format, a...)
	fsrc := "package p\nfunc _() {\n" + src + "\n}\n"
	file, err := parser.ParseFile(token.NewFileSet(), "", fsrc, 0)
	if err != nil {
		log.Fatalf("stmt parse error on %q: %v", src, err)
	}
	return file.Decls[0].(*ast.FuncDecl).Body.List[0]
}

// declf constructs a simple "name := value" declaration, using exprf for its
// value.
func declf(name, format string, a ...interface{}) *Declare {
	return &Declare{name, exprf(format, a...)}
}

// breakf constructs a simple "if cond { break }" statement, using exprf for its
// condition.
func breakf(format string, a ...interface{}) *CondBreak {
	return &CondBreak{expr: exprf(format, a...)}
}

func genBlockRewrite(rule Rule, arch arch, data blockData) *RuleRewrite {
	rr := &RuleRewrite{loc: rule.loc}
	rr.match, rr.cond, rr.result = rule.parse()
	_, _, auxint, aux, s := extract(rr.match) // remove parens, then split

	// check match of control values
	if len(s) < data.controls {
		log.Fatalf("incorrect number of arguments in %s, got %v wanted at least %v", rule, len(s), data.controls)
	}
	controls := s[:data.controls]
	pos := make([]string, data.controls)
	for i, arg := range controls {
		if strings.Contains(arg, "(") {
			// TODO: allow custom names?
			cname := fmt.Sprintf("b.Controls[%v]", i)
			vname := fmt.Sprintf("v_%v", i)
			rr.add(declf(vname, cname))
			p, op := genMatch0(rr, arch, arg, vname, nil, false) // TODO: pass non-nil cnt?
			if op != "" {
				check := fmt.Sprintf("%s.Op == %s", cname, op)
				if rr.check == "" {
					rr.check = check
				} else {
					rr.check = rr.check + " && " + check
				}
			}
			if p == "" {
				p = vname + ".Pos"
			}
			pos[i] = p
		} else {
			rr.add(declf(arg, "b.Controls[%v]", i))
			pos[i] = arg + ".Pos"
		}
	}
	for _, e := range []struct {
		name, field string
	}{
		{auxint, "AuxInt"},
		{aux, "Aux"},
	} {
		if e.name == "" {
			continue
		}
		if !token.IsIdentifier(e.name) || rr.declared(e.name) {
			// code or variable
			rr.add(breakf("b.%s != %s", e.field, e.name))
		} else {
			rr.add(declf(e.name, "b.%s", e.field))
		}
	}
	if rr.cond != "" {
		rr.add(breakf("!(%s)", rr.cond))
	}

	// Rule matches. Generate result.
	outop, _, auxint, aux, t := extract(rr.result) // remove parens, then split
	_, outdata := getBlockInfo(outop, arch)
	if len(t) < outdata.controls {
		log.Fatalf("incorrect number of output arguments in %s, got %v wanted at least %v", rule, len(s), outdata.controls)
	}

	// Check if newsuccs is the same set as succs.
	succs := s[data.controls:]
	newsuccs := t[outdata.controls:]
	m := map[string]bool{}
	for _, succ := range succs {
		if m[succ] {
			log.Fatalf("can't have a repeat successor name %s in %s", succ, rule)
		}
		m[succ] = true
	}
	for _, succ := range newsuccs {
		if !m[succ] {
			log.Fatalf("unknown successor %s in %s", succ, rule)
		}
		delete(m, succ)
	}
	if len(m) != 0 {
		log.Fatalf("unmatched successors %v in %s", m, rule)
	}

	blockName, _ := getBlockInfo(outop, arch)
	var genControls [2]string
	for i, control := range t[:outdata.controls] {
		// Select a source position for any new control values.
		// TODO: does it always make sense to use the source position
		// of the original control values or should we be using the
		// block's source position in some cases?
		newpos := "b.Pos" // default to block's source position
		if i < len(pos) && pos[i] != "" {
			// Use the previous control value's source position.
			newpos = pos[i]
		}

		// Generate a new control value (or copy an existing value).
		genControls[i] = genResult0(rr, arch, control, false, false, newpos)
	}
	switch outdata.controls {
	case 0:
		rr.add(stmtf("b.Reset(%s)", blockName))
	case 1:
		rr.add(stmtf("b.resetWithControl(%s, %s)", blockName, genControls[0]))
	case 2:
		rr.add(stmtf("b.resetWithControl2(%s, %s, %s)", blockName, genControls[0], genControls[1]))
	default:
		log.Fatalf("too many controls: %d", outdata.controls)
	}

	if auxint != "" {
		rr.add(stmtf("b.AuxInt = %s", auxint))
	}
	if aux != "" {
		rr.add(stmtf("b.Aux = %s", aux))
	}

	succChanged := false
	for i := 0; i < len(succs); i++ {
		if succs[i] != newsuccs[i] {
			succChanged = true
		}
	}
	if succChanged {
		if len(succs) != 2 {
			log.Fatalf("changed successors, len!=2 in %s", rule)
		}
		if succs[0] != newsuccs[1] || succs[1] != newsuccs[0] {
			log.Fatalf("can only handle swapped successors in %s", rule)
		}
		rr.add(stmtf("b.swapSuccessors()"))
	}

	if *genLog {
		rr.add(stmtf("logRule(%q)", rule.loc))
	}
	return rr
}

// genMatch returns the variable whose source position should be used for the
// result (or "" if no opinion), and a boolean that reports whether the match can fail.
func genMatch(rr *RuleRewrite, arch arch, match string, pregenTop bool) (pos, checkOp string) {
	cnt := varCount(rr)
	return genMatch0(rr, arch, match, "v", cnt, pregenTop)
}

func genMatch0(rr *RuleRewrite, arch arch, match, v string, cnt map[string]int, pregenTop bool) (pos, checkOp string) {
	if match[0] != '(' || match[len(match)-1] != ')' {
		log.Fatalf("%s: non-compound expr in genMatch0: %q", rr.loc, match)
	}
	op, oparch, typ, auxint, aux, args := parseValue(match, arch, rr.loc)

	checkOp = fmt.Sprintf("Op%s%s", oparch, op.name)

	if op.faultOnNilArg0 || op.faultOnNilArg1 {
		// Prefer the position of an instruction which could fault.
		pos = v + ".Pos"
	}

	for _, e := range []struct {
		name, field string
	}{
		{typ, "Type"},
		{auxint, "AuxInt"},
		{aux, "Aux"},
	} {
		if e.name == "" {
			continue
		}
		if !token.IsIdentifier(e.name) || rr.declared(e.name) {
			// code or variable
			rr.add(breakf("%s.%s != %s", v, e.field, e.name))
		} else {
			rr.add(declf(e.name, "%s.%s", v, e.field))
		}
	}

	commutative := op.commutative
	if commutative {
		if args[0] == args[1] {
			// When we have (Add x x), for any x,
			// even if there are other uses of x besides these two,
			// and even if x is not a variable,
			// we can skip the commutative match.
			commutative = false
		}
		if cnt[args[0]] == 1 && cnt[args[1]] == 1 {
			// When we have (Add x y) with no other uses
			// of x and y in the matching rule and condition,
			// then we can skip the commutative match (Add y x).
			commutative = false
		}
	}

	if !pregenTop {
		// Access last argument first to minimize bounds checks.
		for n := len(args) - 1; n > 0; n-- {
			a := args[n]
			if a == "_" {
				continue
			}
			if !rr.declared(a) && token.IsIdentifier(a) && !(commutative && len(args) == 2) {
				rr.add(declf(a, "%s.Args[%d]", v, n))
				// delete the last argument so it is not reprocessed
				args = args[:n]
			} else {
				rr.add(stmtf("_ = %s.Args[%d]", v, n))
			}
			break
		}
	}
	if commutative && !pregenTop {
		for i := 0; i <= 1; i++ {
			vname := fmt.Sprintf("%s_%d", v, i)
			rr.add(declf(vname, "%s.Args[%d]", v, i))
		}
	}
	var commuteDepth int
	if commutative {
		commuteDepth = rr.commuteDepth
		rr.add(StartCommuteLoop{commuteDepth, v})
		rr.commuteDepth++
	}
	for i, arg := range args {
		if arg == "_" {
			continue
		}
		var rhs string
		if (commutative && i < 2) || pregenTop {
			rhs = fmt.Sprintf("%s_%d", v, i)
		} else {
			rhs = fmt.Sprintf("%s.Args[%d]", v, i)
		}
		if !strings.Contains(arg, "(") {
			// leaf variable
			if rr.declared(arg) {
				// variable already has a definition. Check whether
				// the old definition and the new definition match.
				// For example, (add x x).  Equality is just pointer equality
				// on Values (so cse is important to do before lowering).
				rr.add(breakf("%s != %s", arg, rhs))
			} else {
				if arg != rhs {
					rr.add(declf(arg, "%s", rhs))
				}
			}
			continue
		}
		// compound sexpr
		argname, expr := splitNameExpr(arg)
		if argname == "" {
			argname = fmt.Sprintf("%s_%d", v, i)
		}
		if argname == "b" {
			log.Fatalf("don't name args 'b', it is ambiguous with blocks")
		}

		if argname != rhs {
			rr.add(declf(argname, "%s", rhs))
		}
		bexpr := exprf("%s.Op != addLater", argname)
		rr.add(&CondBreak{expr: bexpr})
		argPos, argCheckOp := genMatch0(rr, arch, expr, argname, cnt, false)
		bexpr.(*ast.BinaryExpr).Y.(*ast.Ident).Name = argCheckOp

		if argPos != "" {
			// Keep the argument in preference to the parent, as the
			// argument is normally earlier in program flow.
			// Keep the argument in preference to an earlier argument,
			// as that prefers the memory argument which is also earlier
			// in the program flow.
			pos = argPos
		}
	}

	if op.argLength == -1 {
		rr.add(breakf("len(%s.Args) != %d", v, len(args)))
	}
	return pos, checkOp
}

func genResult(rr *RuleRewrite, arch arch, result, pos string) {
	move := result[0] == '@'
	if move {
		// parse @block directive
		s := strings.SplitN(result[1:], " ", 2)
		rr.add(stmtf("b = %s", s[0]))
		result = s[1]
	}
	genResult0(rr, arch, result, true, move, pos)
}

func genResult0(rr *RuleRewrite, arch arch, result string, top, move bool, pos string) string {
	// TODO: when generating a constant result, use f.constVal to avoid
	// introducing copies just to clean them up again.
	if result[0] != '(' {
		// variable
		if top {
			// It in not safe in general to move a variable between blocks
			// (and particularly not a phi node).
			// Introduce a copy.
			rr.add(stmtf("v.copyOf(%s)", result))
		}
		return result
	}

	op, oparch, typ, auxint, aux, args := parseValue(result, arch, rr.loc)

	// Find the type of the variable.
	typeOverride := typ != ""
	if typ == "" && op.typ != "" {
		typ = typeName(op.typ)
	}

	v := "v"
	if top && !move {
		rr.add(stmtf("v.reset(Op%s%s)", oparch, op.name))
		if typeOverride {
			rr.add(stmtf("v.Type = %s", typ))
		}
	} else {
		if typ == "" {
			log.Fatalf("sub-expression %s (op=Op%s%s) at %s must have a type", result, oparch, op.name, rr.loc)
		}
		v = fmt.Sprintf("v%d", rr.alloc)
		rr.alloc++
		rr.add(declf(v, "b.NewValue0(%s, Op%s%s, %s)", pos, oparch, op.name, typ))
		if move && top {
			// Rewrite original into a copy
			rr.add(stmtf("v.copyOf(%s)", v))
		}
	}

	if auxint != "" {
		rr.add(stmtf("%s.AuxInt = %s", v, auxint))
	}
	if aux != "" {
		rr.add(stmtf("%s.Aux = %s", v, aux))
	}
	all := new(strings.Builder)
	for i, arg := range args {
		x := genResult0(rr, arch, arg, false, move, pos)
		if i > 0 {
			all.WriteString(", ")
		}
		all.WriteString(x)
	}
	switch len(args) {
	case 0:
	case 1:
		rr.add(stmtf("%s.AddArg(%s)", v, all.String()))
	default:
		rr.add(stmtf("%s.AddArg%d(%s)", v, len(args), all.String()))
	}
	return v
}

func split(s string) []string {
	var r []string

outer:
	for s != "" {
		d := 0               // depth of ({[<
		var open, close byte // opening and closing markers ({[< or )}]>
		nonsp := false       // found a non-space char so far
		for i := 0; i < len(s); i++ {
			switch {
			case d == 0 && s[i] == '(':
				open, close = '(', ')'
				d++
			case d == 0 && s[i] == '<':
				open, close = '<', '>'
				d++
			case d == 0 && s[i] == '[':
				open, close = '[', ']'
				d++
			case d == 0 && s[i] == '{':
				open, close = '{', '}'
				d++
			case d == 0 && (s[i] == ' ' || s[i] == '\t'):
				if nonsp {
					r = append(r, strings.TrimSpace(s[:i]))
					s = s[i:]
					continue outer
				}
			case d > 0 && s[i] == open:
				d++
			case d > 0 && s[i] == close:
				d--
			default:
				nonsp = true
			}
		}
		if d != 0 {
			log.Fatalf("imbalanced expression: %q", s)
		}
		if nonsp {
			r = append(r, strings.TrimSpace(s))
		}
		break
	}
	return r
}

// isBlock reports whether this op is a block opcode.
func isBlock(name string, arch arch) bool {
	for _, b := range genericBlocks {
		if b.name == name {
			return true
		}
	}
	for _, b := range arch.blocks {
		if b.name == name {
			return true
		}
	}
	return false
}

func extract(val string) (op, typ, auxint, aux string, args []string) {
	val = val[1 : len(val)-1] // remove ()

	// Split val up into regions.
	// Split by spaces/tabs, except those contained in (), {}, [], or <>.
	s := split(val)

	// Extract restrictions and args.
	op = s[0]
	for _, a := range s[1:] {
		switch a[0] {
		case '<':
			typ = a[1 : len(a)-1] // remove <>
		case '[':
			auxint = a[1 : len(a)-1] // remove []
		case '{':
			aux = a[1 : len(a)-1] // remove {}
		default:
			args = append(args, a)
		}
	}
	return
}

// parseValue parses a parenthesized value from a rule.
// The value can be from the match or the result side.
// It returns the op and unparsed strings for typ, auxint, and aux restrictions and for all args.
// oparch is the architecture that op is located in, or "" for generic.
func parseValue(val string, arch arch, loc string) (op opData, oparch, typ, auxint, aux string, args []string) {
	// Resolve the op.
	var s string
	s, typ, auxint, aux, args = extract(val)

	// match reports whether x is a good op to select.
	// If strict is true, rule generation might succeed.
	// If strict is false, rule generation has failed,
	// but we're trying to generate a useful error.
	// Doing strict=true then strict=false allows
	// precise op matching while retaining good error messages.
	match := func(x opData, strict bool, archname string) bool {
		if x.name != s {
			return false
		}
		if x.argLength != -1 && int(x.argLength) != len(args) && (len(args) != 1 || args[0] != "...") {
			if strict {
				return false
			}
			log.Printf("%s: op %s (%s) should have %d args, has %d", loc, s, archname, x.argLength, len(args))
		}
		return true
	}

	for _, x := range genericOps {
		if match(x, true, "generic") {
			op = x
			break
		}
	}
	for _, x := range arch.ops {
		if arch.name != "generic" && match(x, true, arch.name) {
			if op.name != "" {
				log.Fatalf("%s: matches for op %s found in both generic and %s", loc, op.name, arch.name)
			}
			op = x
			oparch = arch.name
			break
		}
	}

	if op.name == "" {
		// Failed to find the op.
		// Run through everything again with strict=false
		// to generate useful diagnosic messages before failing.
		for _, x := range genericOps {
			match(x, false, "generic")
		}
		for _, x := range arch.ops {
			match(x, false, arch.name)
		}
		log.Fatalf("%s: unknown op %s", loc, s)
	}

	// Sanity check aux, auxint.
	if auxint != "" && !opHasAuxInt(op) {
		log.Fatalf("%s: op %s %s can't have auxint", loc, op.name, op.aux)
	}
	if aux != "" && !opHasAux(op) {
		log.Fatalf("%s: op %s %s can't have aux", loc, op.name, op.aux)
	}
	return
}

func opHasAuxInt(op opData) bool {
	switch op.aux {
	case "Bool", "Int8", "Int16", "Int32", "Int64", "Int128", "Float32", "Float64", "SymOff", "SymValAndOff", "TypSize", "ARM64BitField":
		return true
	}
	return false
}

func opHasAux(op opData) bool {
	switch op.aux {
	case "String", "Sym", "SymOff", "SymValAndOff", "Typ", "TypSize", "CCop", "ArchSpecific":
		return true
	}
	return false
}

// splitNameExpr splits s-expr arg, possibly prefixed by "name:",
// into name and the unprefixed expression.
// For example, "x:(Foo)" yields "x", "(Foo)",
// and "(Foo)" yields "", "(Foo)".
func splitNameExpr(arg string) (name, expr string) {
	colon := strings.Index(arg, ":")
	if colon < 0 {
		return "", arg
	}
	openparen := strings.Index(arg, "(")
	if openparen < 0 {
		log.Fatalf("splitNameExpr(%q): colon but no open parens", arg)
	}
	if colon > openparen {
		// colon is inside the parens, such as in "(Foo x:(Bar))".
		return "", arg
	}
	return arg[:colon], arg[colon+1:]
}

func getBlockInfo(op string, arch arch) (name string, data blockData) {
	for _, b := range genericBlocks {
		if b.name == op {
			return "Block" + op, b
		}
	}
	for _, b := range arch.blocks {
		if b.name == op {
			return "Block" + arch.name + op, b
		}
	}
	log.Fatalf("could not find block data for %s", op)
	panic("unreachable")
}

// typeName returns the string to use to generate a type.
func typeName(typ string) string {
	if typ[0] == '(' {
		ts := strings.Split(typ[1:len(typ)-1], ",")
		if len(ts) != 2 {
			log.Fatalf("Tuple expect 2 arguments")
		}
		return "types.NewTuple(" + typeName(ts[0]) + ", " + typeName(ts[1]) + ")"
	}
	switch typ {
	case "Flags", "Mem", "Void", "Int128":
		return "types.Type" + typ
	default:
		return "typ." + typ
	}
}

// unbalanced reports whether there are a different number of ( and ) in the string.
func unbalanced(s string) bool {
	balance := 0
	for _, c := range s {
		if c == '(' {
			balance++
		} else if c == ')' {
			balance--
		}
	}
	return balance != 0
}

// findAllOpcode is a function to find the opcode portion of s-expressions.
var findAllOpcode = regexp.MustCompile(`[(](\w+[|])+\w+[)]`).FindAllStringIndex

// excludeFromExpansion reports whether the substring s[idx[0]:idx[1]] in a rule
// should be disregarded as a candidate for | expansion.
// It uses simple syntactic checks to see whether the substring
// is inside an AuxInt expression or inside the && conditions.
func excludeFromExpansion(s string, idx []int) bool {
	left := s[:idx[0]]
	if strings.LastIndexByte(left, '[') > strings.LastIndexByte(left, ']') {
		// Inside an AuxInt expression.
		return true
	}
	right := s[idx[1]:]
	if strings.Contains(left, "&&") && strings.Contains(right, "->") {
		// Inside && conditions.
		return true
	}
	return false
}

// expandOr converts a rule into multiple rules by expanding | ops.
func expandOr(r string) []string {
	// Find every occurrence of |-separated things.
	// They look like MOV(B|W|L|Q|SS|SD)load or MOV(Q|L)loadidx(1|8).
	// Generate rules selecting one case from each |-form.

	// Count width of |-forms.  They must match.
	n := 1
	for _, idx := range findAllOpcode(r, -1) {
		if excludeFromExpansion(r, idx) {
			continue
		}
		s := r[idx[0]:idx[1]]
		c := strings.Count(s, "|") + 1
		if c == 1 {
			continue
		}
		if n > 1 && n != c {
			log.Fatalf("'|' count doesn't match in %s: both %d and %d\n", r, n, c)
		}
		n = c
	}
	if n == 1 {
		// No |-form in this rule.
		return []string{r}
	}
	// Build each new rule.
	res := make([]string, n)
	for i := 0; i < n; i++ {
		buf := new(strings.Builder)
		x := 0
		for _, idx := range findAllOpcode(r, -1) {
			if excludeFromExpansion(r, idx) {
				continue
			}
			buf.WriteString(r[x:idx[0]])              // write bytes we've skipped over so far
			s := r[idx[0]+1 : idx[1]-1]               // remove leading "(" and trailing ")"
			buf.WriteString(strings.Split(s, "|")[i]) // write the op component for this rule
			x = idx[1]                                // note that we've written more bytes
		}
		buf.WriteString(r[x:])
		res[i] = buf.String()
	}
	return res
}

// varCount returns a map which counts the number of occurrences of
// Value variables in the s-expression rr.match and the Go expression rr.cond.
func varCount(rr *RuleRewrite) map[string]int {
	cnt := map[string]int{}
	varCount1(rr.loc, rr.match, cnt)
	if rr.cond != "" {
		expr, err := parser.ParseExpr(rr.cond)
		if err != nil {
			log.Fatalf("%s: failed to parse cond %q: %v", rr.loc, rr.cond, err)
		}
		ast.Inspect(expr, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				cnt[id.Name]++
			}
			return true
		})
	}
	return cnt
}

func varCount1(loc, m string, cnt map[string]int) {
	if m[0] == '<' || m[0] == '[' || m[0] == '{' {
		return
	}
	if token.IsIdentifier(m) {
		cnt[m]++
		return
	}
	// Split up input.
	name, expr := splitNameExpr(m)
	if name != "" {
		cnt[name]++
	}
	if expr[0] != '(' || expr[len(expr)-1] != ')' {
		log.Fatalf("%s: non-compound expr in varCount1: %q", loc, expr)
	}
	s := split(expr[1 : len(expr)-1])
	for _, arg := range s[1:] {
		varCount1(loc, arg, cnt)
	}
}

// normalizeWhitespace replaces 2+ whitespace sequences with a single space.
func normalizeWhitespace(x string) string {
	x = strings.Join(strings.Fields(x), " ")
	x = strings.Replace(x, "( ", "(", -1)
	x = strings.Replace(x, " )", ")", -1)
	x = strings.Replace(x, "[ ", "[", -1)
	x = strings.Replace(x, " ]", "]", -1)
	x = strings.Replace(x, ")->", ") ->", -1)
	return x
}

// opIsCommutative reports whether op s is commutative.
func opIsCommutative(op string, arch arch) bool {
	for _, x := range genericOps {
		if op == x.name {
			if x.commutative {
				return true
			}
			break
		}
	}
	if arch.name != "generic" {
		for _, x := range arch.ops {
			if op == x.name {
				if x.commutative {
					return true
				}
				break
			}
		}
	}
	return false
}

func normalizeMatch(m string, arch arch) string {
	if token.IsIdentifier(m) {
		return m
	}
	op, typ, auxint, aux, args := extract(m)
	if opIsCommutative(op, arch) {
		if args[1] < args[0] {
			args[0], args[1] = args[1], args[0]
		}
	}
	s := new(strings.Builder)
	fmt.Fprintf(s, "%s <%s> [%s] {%s}", op, typ, auxint, aux)
	for _, arg := range args {
		prefix, expr := splitNameExpr(arg)
		fmt.Fprint(s, " ", prefix, normalizeMatch(expr, arch))
	}
	return s.String()
}

func parseEllipsisRules(rules []Rule, arch arch) (newop string, ok bool) {
	if len(rules) != 1 {
		for _, r := range rules {
			if strings.Contains(r.rule, "...") {
				log.Fatalf("%s: found ellipsis in rule, but there are other rules with the same op", r.loc)
			}
		}
		return "", false
	}
	rule := rules[0]
	match, cond, result := rule.parse()
	if cond != "" || !isEllipsisValue(match) || !isEllipsisValue(result) {
		if strings.Contains(rule.rule, "...") {
			log.Fatalf("%s: found ellipsis in non-ellipsis rule", rule.loc)
		}
		checkEllipsisRuleCandidate(rule, arch)
		return "", false
	}
	op, oparch, _, _, _, _ := parseValue(result, arch, rule.loc)
	return fmt.Sprintf("Op%s%s", oparch, op.name), true
}

// isEllipsisValue reports whether s is of the form (OpX ...).
func isEllipsisValue(s string) bool {
	if len(s) < 2 || s[0] != '(' || s[len(s)-1] != ')' {
		return false
	}
	c := split(s[1 : len(s)-1])
	if len(c) != 2 || c[1] != "..." {
		return false
	}
	return true
}

func checkEllipsisRuleCandidate(rule Rule, arch arch) {
	match, cond, result := rule.parse()
	if cond != "" {
		return
	}
	op, _, _, auxint, aux, args := parseValue(match, arch, rule.loc)
	var auxint2, aux2 string
	var args2 []string
	var usingCopy string
	if result[0] != '(' {
		// Check for (Foo x) -> x, which can be converted to (Foo ...) -> (Copy ...).
		args2 = []string{result}
		usingCopy = " using Copy"
	} else {
		_, _, _, auxint2, aux2, args2 = parseValue(result, arch, rule.loc)
	}
	// Check that all restrictions in match are reproduced exactly in result.
	if aux != aux2 || auxint != auxint2 || len(args) != len(args2) {
		return
	}
	for i := range args {
		if args[i] != args2[i] {
			return
		}
	}
	switch {
	case opHasAux(op) && aux == "" && aux2 == "":
		fmt.Printf("%s: rule silently zeros aux, either copy aux or explicitly zero\n", rule.loc)
	case opHasAuxInt(op) && auxint == "" && auxint2 == "":
		fmt.Printf("%s: rule silently zeros auxint, either copy auxint or explicitly zero\n", rule.loc)
	default:
		fmt.Printf("%s: possible ellipsis rule candidate%s: %q\n", rule.loc, usingCopy, rule.rule)
	}
}

func opByName(arch arch, name string) opData {
	name = name[2:]
	for _, x := range genericOps {
		if name == x.name {
			return x
		}
	}
	if arch.name != "generic" {
		name = name[len(arch.name):]
		for _, x := range arch.ops {
			if name == x.name {
				return x
			}
		}
	}
	log.Fatalf("failed to find op named %s in arch %s", name, arch.name)
	panic("unreachable")
}
