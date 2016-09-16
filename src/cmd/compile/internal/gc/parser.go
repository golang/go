// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// The recursive-descent parser is built around a slighty modified grammar
// of Go to accommodate for the constraints imposed by strict one token look-
// ahead, and for better error handling. Subsequent checks of the constructed
// syntax tree restrict the language accepted by the compiler to proper Go.
//
// Semicolons are inserted by the lexer. The parser uses one-token look-ahead
// to handle optional commas and semicolons before a closing ) or } .

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const trace = false // if set, parse tracing can be enabled with -x

// oldParseFile parses a single Go source file.
func oldParseFile(infile string) {
	f, err := os.Open(infile)
	if err != nil {
		fmt.Printf("open %s: %v\n", infile, err)
		errorexit()
	}
	defer f.Close()
	bin := bufio.NewReader(f)

	// Skip initial BOM if present.
	if r, _, _ := bin.ReadRune(); r != BOM {
		bin.UnreadRune()
	}
	newparser(bin, nil).file()
}

type parser struct {
	lexer
	fnest  int    // function nesting level (for error handling)
	xnest  int    // expression nesting level (for complit ambiguity resolution)
	indent []byte // tracing support
}

// newparser returns a new parser ready to parse from src.
// indent is the initial indentation for tracing output.
func newparser(src *bufio.Reader, indent []byte) *parser {
	var p parser
	p.bin = src
	p.indent = indent
	p.next()
	return &p
}

func (p *parser) got(tok int32) bool {
	if p.tok == tok {
		p.next()
		return true
	}
	return false
}

func (p *parser) want(tok int32) {
	if !p.got(tok) {
		p.syntax_error("expecting " + tokstring(tok))
		p.advance()
	}
}

// ----------------------------------------------------------------------------
// Syntax error handling

func (p *parser) syntax_error(msg string) {
	if trace && Debug['x'] != 0 {
		defer p.trace("syntax_error (" + msg + ")")()
	}

	if p.tok == EOF && nerrors > 0 {
		return // avoid meaningless follow-up errors
	}

	// add punctuation etc. as needed to msg
	switch {
	case msg == "":
		// nothing to do
	case strings.HasPrefix(msg, "in"), strings.HasPrefix(msg, "at"), strings.HasPrefix(msg, "after"):
		msg = " " + msg
	case strings.HasPrefix(msg, "expecting"):
		msg = ", " + msg
	default:
		// plain error - we don't care about current token
		yyerror("syntax error: %s", msg)
		return
	}

	// determine token string
	var tok string
	switch p.tok {
	case LNAME:
		if p.sym_ != nil && p.sym_.Name != "" {
			tok = p.sym_.Name
		} else {
			tok = "name"
		}
	case LLITERAL:
		if litbuf == "" {
			litbuf = "literal " + lexbuf.String()
		}
		tok = litbuf
	case LOPER:
		tok = goopnames[p.op]
	case LASOP:
		tok = goopnames[p.op] + "="
	case LINCOP:
		tok = goopnames[p.op] + goopnames[p.op]
	default:
		tok = tokstring(p.tok)
	}

	yyerror("syntax error: unexpected %s", tok+msg)
}

// Like syntax_error, but reports error at given line rather than current lexer line.
func (p *parser) syntax_error_at(lno int32, msg string) {
	defer func(lno int32) {
		lineno = lno
	}(lineno)
	lineno = lno
	p.syntax_error(msg)
}

// The stoplist contains keywords that start a statement.
// They are good synchronization points in case of syntax
// errors and (usually) shouldn't be skipped over.
var stoplist = map[int32]bool{
	LBREAK:    true,
	LCONST:    true,
	LCONTINUE: true,
	LDEFER:    true,
	LFALL:     true,
	LFOR:      true,
	LFUNC:     true,
	LGO:       true,
	LGOTO:     true,
	LIF:       true,
	LRETURN:   true,
	LSELECT:   true,
	LSWITCH:   true,
	LTYPE:     true,
	LVAR:      true,
}

// Advance consumes tokens until it finds a token of the stop- or followlist.
// The stoplist is only considered if we are inside a function (p.fnest > 0).
// The followlist is the list of valid tokens that can follow a production;
// if it is empty, exactly one token is consumed to ensure progress.
func (p *parser) advance(followlist ...int32) {
	if len(followlist) == 0 {
		p.next()
		return
	}
	for p.tok != EOF {
		if p.fnest > 0 && stoplist[p.tok] {
			return
		}
		for _, follow := range followlist {
			if p.tok == follow {
				return
			}
		}
		p.next()
	}
}

func tokstring(tok int32) string {
	switch tok {
	case EOF:
		return "EOF"
	case ',':
		return "comma"
	case ';':
		return "semicolon or newline"
	}
	if 0 <= tok && tok < 128 {
		// get invisibles properly backslashed
		s := strconv.QuoteRune(tok)
		if n := len(s); n > 0 && s[0] == '\'' && s[n-1] == '\'' {
			s = s[1 : n-1]
		}
		return s
	}
	if s := tokstrings[tok]; s != "" {
		return s
	}
	// catchall
	return fmt.Sprintf("tok-%v", tok)
}

var tokstrings = map[int32]string{
	LNAME:    "NAME",
	LLITERAL: "LITERAL",

	LOPER:  "op",
	LASOP:  "op=",
	LINCOP: "opop",

	LCOLAS: ":=",
	LCOMM:  "<-",
	LDDD:   "...",

	LBREAK:     "break",
	LCASE:      "case",
	LCHAN:      "chan",
	LCONST:     "const",
	LCONTINUE:  "continue",
	LDEFAULT:   "default",
	LDEFER:     "defer",
	LELSE:      "else",
	LFALL:      "fallthrough",
	LFOR:       "for",
	LFUNC:      "func",
	LGO:        "go",
	LGOTO:      "goto",
	LIF:        "if",
	LIMPORT:    "import",
	LINTERFACE: "interface",
	LMAP:       "map",
	LPACKAGE:   "package",
	LRANGE:     "range",
	LRETURN:    "return",
	LSELECT:    "select",
	LSTRUCT:    "struct",
	LSWITCH:    "switch",
	LTYPE:      "type",
	LVAR:       "var",
}

// usage: defer p.trace(msg)()
func (p *parser) trace(msg string) func() {
	fmt.Printf("%5d: %s%s (\n", lineno, p.indent, msg)
	const tab = ". "
	p.indent = append(p.indent, tab...)
	return func() {
		p.indent = p.indent[:len(p.indent)-len(tab)]
		if x := recover(); x != nil {
			panic(x) // skip print_trace
		}
		fmt.Printf("%5d: %s)\n", lineno, p.indent)
	}
}

// ----------------------------------------------------------------------------
// Parsing package files
//
// Parse methods are annotated with matching Go productions as appropriate.
// The annotations are intended as guidelines only since a single Go grammar
// rule may be covered by multiple parse methods and vice versa.

// SourceFile = PackageClause ";" { ImportDecl ";" } { TopLevelDecl ";" } .
func (p *parser) file() {
	if trace && Debug['x'] != 0 {
		defer p.trace("file")()
	}

	p.package_()
	p.want(';')

	for p.tok == LIMPORT {
		p.import_()
		p.want(';')
	}

	xtop = append(xtop, p.xdcl_list()...)

	p.want(EOF)
}

// PackageClause = "package" PackageName .
// PackageName   = identifier .
func (p *parser) package_() {
	if trace && Debug['x'] != 0 {
		defer p.trace("package_")()
	}

	if !p.got(LPACKAGE) {
		p.syntax_error("package statement must be first")
		errorexit()
	}
	mkpackage(p.sym().Name)
}

// ImportDecl = "import" ( ImportSpec | "(" { ImportSpec ";" } ")" ) .
func (p *parser) import_() {
	if trace && Debug['x'] != 0 {
		defer p.trace("import_")()
	}

	p.want(LIMPORT)
	if p.got('(') {
		for p.tok != EOF && p.tok != ')' {
			p.importdcl()
			if !p.osemi(')') {
				break
			}
		}
		p.want(')')
	} else {
		p.importdcl()
	}
}

// ImportSpec = [ "." | PackageName ] ImportPath .
// ImportPath = string_lit .
func (p *parser) importdcl() {
	if trace && Debug['x'] != 0 {
		defer p.trace("importdcl")()
	}

	var my *Sym
	switch p.tok {
	case LNAME:
		// import with given name
		my = p.sym()

	case '.':
		// import into my name space
		my = lookup(".")
		p.next()
	}

	if p.tok != LLITERAL {
		p.syntax_error("missing import path; require quoted string")
		p.advance(';', ')')
		return
	}

	line := lineno

	// We need to clear importpkg before calling p.next(),
	// otherwise it will affect lexlineno.
	// TODO(mdempsky): Fix this clumsy API.
	importfile(&p.val, p.indent)
	ipkg := importpkg
	importpkg = nil

	p.next()
	if ipkg == nil {
		if nerrors == 0 {
			Fatalf("phase error in import")
		}
		return
	}

	ipkg.Direct = true

	if my == nil {
		my = lookup(ipkg.Name)
	}

	pack := nod(OPACK, nil, nil)
	pack.Sym = my
	pack.Name.Pkg = ipkg
	pack.Lineno = line

	if strings.HasPrefix(my.Name, ".") {
		importdot(ipkg, pack)
		return
	}
	if my.Name == "init" {
		lineno = line
		yyerror("cannot import package as init - init must be a func")
		return
	}
	if my.Name == "_" {
		return
	}
	if my.Def != nil {
		lineno = line
		redeclare(my, "as imported package name")
	}
	my.Def = pack
	my.Lastlineno = line
	my.Block = 1 // at top level
}

// Declaration = ConstDecl | TypeDecl | VarDecl .
// ConstDecl   = "const" ( ConstSpec | "(" { ConstSpec ";" } ")" ) .
// TypeDecl    = "type" ( TypeSpec | "(" { TypeSpec ";" } ")" ) .
// VarDecl     = "var" ( VarSpec | "(" { VarSpec ";" } ")" ) .
func (p *parser) common_dcl() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("common_dcl")()
	}

	var dcl func() []*Node
	switch p.tok {
	case LVAR:
		dcl = p.vardcl

	case LCONST:
		iota_ = 0
		dcl = p.constdcl

	case LTYPE:
		dcl = p.typedcl

	default:
		panic("unreachable")
	}

	p.next()
	var s []*Node
	if p.got('(') {
		for p.tok != EOF && p.tok != ')' {
			s = append(s, dcl()...)
			if !p.osemi(')') {
				break
			}
		}
		p.want(')')
	} else {
		s = dcl()
	}

	iota_ = -100000
	lastconst = nil

	return s
}

// VarSpec = IdentifierList ( Type [ "=" ExpressionList ] | "=" ExpressionList ) .
func (p *parser) vardcl() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("vardcl")()
	}

	names := p.dcl_name_list()
	var typ *Node
	var exprs []*Node
	if p.got('=') {
		exprs = p.expr_list()
	} else {
		typ = p.ntype()
		if p.got('=') {
			exprs = p.expr_list()
		}
	}

	return variter(names, typ, exprs)
}

// ConstSpec = IdentifierList [ [ Type ] "=" ExpressionList ] .
func (p *parser) constdcl() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("constdcl")()
	}

	names := p.dcl_name_list()
	var typ *Node
	var exprs []*Node
	if p.tok != EOF && p.tok != ';' && p.tok != ')' {
		typ = p.try_ntype()
		if p.got('=') {
			exprs = p.expr_list()
		}
	}

	return constiter(names, typ, exprs)
}

// TypeSpec = identifier Type .
func (p *parser) typedcl() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("typedcl")()
	}

	name := typedcl0(p.sym())

	typ := p.try_ntype()
	// handle case where type is missing
	if typ == nil {
		p.syntax_error("in type declaration")
		p.advance(';', ')')
	}

	return []*Node{typedcl1(name, typ, true)}
}

// SimpleStmt = EmptyStmt | ExpressionStmt | SendStmt | IncDecStmt | Assignment | ShortVarDecl .
//
// simple_stmt may return missing_stmt if labelOk is set.
func (p *parser) simple_stmt(labelOk, rangeOk bool) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("simple_stmt")()
	}

	if rangeOk && p.got(LRANGE) {
		// LRANGE expr
		r := nod(ORANGE, nil, p.expr())
		r.Etype = 0 // := flag
		return r
	}

	lhs := p.expr_list()

	if len(lhs) == 1 && p.tok != '=' && p.tok != LCOLAS && p.tok != LRANGE {
		// expr
		lhs := lhs[0]
		switch p.tok {
		case LASOP:
			// expr LASOP expr
			op := p.op
			p.next()
			rhs := p.expr()

			stmt := nod(OASOP, lhs, rhs)
			stmt.Etype = EType(op) // rathole to pass opcode
			return stmt

		case LINCOP:
			// expr LINCOP
			p.next()

			stmt := nod(OASOP, lhs, nodintconst(1))
			stmt.Implicit = true
			stmt.Etype = EType(p.op)
			return stmt

		case ':':
			// labelname ':' stmt
			if labelOk {
				// If we have a labelname, it was parsed by operand
				// (calling p.name()) and given an ONAME, ONONAME, OTYPE, OPACK, or OLITERAL node.
				// We only have a labelname if there is a symbol (was issue 14006).
				switch lhs.Op {
				case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
					if lhs.Sym != nil {
						lhs = newname(lhs.Sym)
						break
					}
					fallthrough
				default:
					p.syntax_error("expecting semicolon or newline or }")
					// we already progressed, no need to advance
				}
				lhs := nod(OLABEL, lhs, nil)
				lhs.Sym = dclstack // context, for goto restrictions
				p.next()           // consume ':' after making label node for correct lineno
				return p.labeled_stmt(lhs)
			}
			fallthrough

		default:
			// expr
			// Since a bare name used as an expression is an error,
			// introduce a wrapper node where necessary to give the
			// correct line.
			return wrapname(lhs)
		}
	}

	// expr_list
	switch p.tok {
	case '=':
		p.next()
		if rangeOk && p.got(LRANGE) {
			// expr_list '=' LRANGE expr
			r := nod(ORANGE, nil, p.expr())
			r.List.Set(lhs)
			r.Etype = 0 // := flag
			return r
		}

		// expr_list '=' expr_list
		rhs := p.expr_list()

		if len(lhs) == 1 && len(rhs) == 1 {
			// simple
			return nod(OAS, lhs[0], rhs[0])
		}
		// multiple
		stmt := nod(OAS2, nil, nil)
		stmt.List.Set(lhs)
		stmt.Rlist.Set(rhs)
		return stmt

	case LCOLAS:
		lno := lineno
		p.next()

		if rangeOk && p.got(LRANGE) {
			// expr_list LCOLAS LRANGE expr
			r := nod(ORANGE, nil, p.expr())
			r.List.Set(lhs)
			r.Colas = true
			colasdefn(lhs, r)
			return r
		}

		// expr_list LCOLAS expr_list
		rhs := p.expr_list()

		if rhs[0].Op == OTYPESW {
			ts := nod(OTYPESW, nil, rhs[0].Right)
			if len(rhs) > 1 {
				yyerror("expr.(type) must be alone in list")
			}
			if len(lhs) > 1 {
				yyerror("argument count mismatch: %d = %d", len(lhs), 1)
			} else if (lhs[0].Op != ONAME && lhs[0].Op != OTYPE && lhs[0].Op != ONONAME && (lhs[0].Op != OLITERAL || lhs[0].Name == nil)) || isblank(lhs[0]) {
				yyerror("invalid variable name %v in type switch", lhs[0])
			} else {
				ts.Left = dclname(lhs[0].Sym)
			} // it's a colas, so must not re-use an oldname
			return ts
		}
		return colas(lhs, rhs, lno)

	default:
		p.syntax_error("expecting := or = or comma")
		p.advance(';', '}')
		return nil
	}
}

// LabeledStmt = Label ":" Statement .
// Label       = identifier .
func (p *parser) labeled_stmt(label *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("labeled_stmt")()
	}

	var ls *Node // labeled statement
	if p.tok != '}' && p.tok != EOF {
		ls = p.stmt()
		if ls == missing_stmt {
			// report error at line of ':' token
			p.syntax_error_at(label.Lineno, "missing statement after label")
			// we are already at the end of the labeled statement - no need to advance
			return missing_stmt
		}
	}

	label.Name.Defn = ls
	l := []*Node{label}
	if ls != nil {
		if ls.Op == OBLOCK && ls.Ninit.Len() == 0 {
			l = append(l, ls.List.Slice()...)
		} else {
			l = append(l, ls)
		}
	}
	return liststmt(l)
}

// case_ parses a superset of switch and select statement cases.
// Later checks restrict the syntax to valid forms.
//
// ExprSwitchCase = "case" ExpressionList | "default" .
// TypeSwitchCase = "case" TypeList | "default" .
// TypeList       = Type { "," Type } .
// CommCase       = "case" ( SendStmt | RecvStmt ) | "default" .
// RecvStmt       = [ ExpressionList "=" | IdentifierList ":=" ] RecvExpr .
// RecvExpr       = Expression .
func (p *parser) case_(tswitch *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("case_")()
	}

	switch p.tok {
	case LCASE:
		p.next()
		cases := p.expr_list() // expr_or_type_list
		switch p.tok {
		case ':':
			// LCASE expr_or_type_list ':'

			// will be converted to OCASE
			// right will point to next case
			// done in casebody()
			markdcl() // matching popdcl in caseblock
			stmt := nod(OXCASE, nil, nil)
			stmt.List.Set(cases)
			if tswitch != nil {
				if n := tswitch.Left; n != nil {
					// type switch - declare variable
					nn := newname(n.Sym)
					declare(nn, dclcontext)
					stmt.Rlist.Set1(nn)

					// keep track of the instances for reporting unused
					nn.Name.Defn = tswitch
				}
			}

			p.next() // consume ':' after declaring type switch var for correct lineno
			return stmt

		case '=':
			// LCASE expr_or_type_list '=' expr ':'
			p.next()
			rhs := p.expr()

			// will be converted to OCASE
			// right will point to next case
			// done in casebody()
			markdcl() // matching popdcl in caseblock
			stmt := nod(OXCASE, nil, nil)
			var n *Node
			if len(cases) == 1 {
				n = nod(OAS, cases[0], rhs)
			} else {
				n = nod(OAS2, nil, nil)
				n.List.Set(cases)
				n.Rlist.Set1(rhs)
			}
			stmt.List.Set1(n)

			p.want(':') // consume ':' after declaring select cases for correct lineno
			return stmt

		case LCOLAS:
			// LCASE expr_or_type_list LCOLAS expr ':'
			lno := lineno
			p.next()
			rhs := p.expr()

			// will be converted to OCASE
			// right will point to next case
			// done in casebody()
			markdcl() // matching popdcl in caseblock
			stmt := nod(OXCASE, nil, nil)
			stmt.List.Set1(colas(cases, []*Node{rhs}, lno))

			p.want(':') // consume ':' after declaring select cases for correct lineno
			return stmt

		default:
			markdcl()                     // for matching popdcl in caseblock
			stmt := nod(OXCASE, nil, nil) // don't return nil
			p.syntax_error("expecting := or = or : or comma")
			p.advance(LCASE, LDEFAULT, '}')
			return stmt
		}

	case LDEFAULT:
		// LDEFAULT ':'
		p.next()

		markdcl() // matching popdcl in caseblock
		stmt := nod(OXCASE, nil, nil)
		if tswitch != nil {
			if n := tswitch.Left; n != nil {
				// type switch - declare variable
				nn := newname(n.Sym)
				declare(nn, dclcontext)
				stmt.Rlist.Set1(nn)

				// keep track of the instances for reporting unused
				nn.Name.Defn = tswitch
			}
		}

		p.want(':') // consume ':' after declaring type switch var for correct lineno
		return stmt

	default:
		markdcl()                     // matching popdcl in caseblock
		stmt := nod(OXCASE, nil, nil) // don't return nil
		p.syntax_error("expecting case or default or }")
		p.advance(LCASE, LDEFAULT, '}')
		return stmt
	}
}

// Block         = "{" StatementList "}" .
// StatementList = { Statement ";" } .
func (p *parser) compound_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("compound_stmt")()
	}

	markdcl()
	p.want('{')
	l := p.stmt_list()
	p.want('}')
	popdcl()

	if len(l) == 0 {
		return nod(OEMPTY, nil, nil)
	}
	return liststmt(l)
}

// caseblock parses a superset of switch and select clauses.
//
// ExprCaseClause = ExprSwitchCase ":" StatementList .
// TypeCaseClause = TypeSwitchCase ":" StatementList .
// CommClause     = CommCase ":" StatementList .
func (p *parser) caseblock(tswitch *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("caseblock")()
	}

	stmt := p.case_(tswitch) // does markdcl
	stmt.Xoffset = int64(block)
	stmt.Nbody.Set(p.stmt_list())

	popdcl()

	return stmt
}

// caseblock_list parses a superset of switch and select clause lists.
func (p *parser) caseblock_list(tswitch *Node) (l []*Node) {
	if trace && Debug['x'] != 0 {
		defer p.trace("caseblock_list")()
	}

	if !p.got('{') {
		p.syntax_error("missing { after switch clause")
		p.advance(LCASE, LDEFAULT, '}')
	}

	for p.tok != EOF && p.tok != '}' {
		l = append(l, p.caseblock(tswitch))
	}
	p.want('}')
	return
}

// loop_body parses if and for statement bodies.
func (p *parser) loop_body(context string) []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("loop_body")()
	}

	markdcl()
	if !p.got('{') {
		p.syntax_error("missing { after " + context)
		p.advance(LNAME, '}')
	}

	body := p.stmt_list()
	popdcl()
	p.want('}')

	return body
}

// for_header parses the header portion of a for statement.
//
// ForStmt   = "for" [ Condition | ForClause | RangeClause ] Block .
// Condition = Expression .
func (p *parser) for_header() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("for_header")()
	}

	init, cond, post := p.header(true)

	if init != nil || post != nil {
		// init ; test ; incr
		if post != nil && post.Colas {
			yyerror("cannot declare in the for-increment")
		}
		h := nod(OFOR, nil, nil)
		if init != nil {
			h.Ninit.Set1(init)
		}
		h.Left = cond
		h.Right = post
		return h
	}

	if cond != nil && cond.Op == ORANGE {
		// range_stmt - handled by pexpr
		return cond
	}

	// normal test
	h := nod(OFOR, nil, nil)
	h.Left = cond
	return h
}

func (p *parser) for_body() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("for_body")()
	}

	stmt := p.for_header()
	body := p.loop_body("for clause")

	stmt.Nbody.Append(body...)
	return stmt
}

// ForStmt = "for" [ Condition | ForClause | RangeClause ] Block .
func (p *parser) for_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("for_stmt")()
	}

	p.want(LFOR)
	markdcl()
	body := p.for_body()
	popdcl()

	return body
}

// header parses a combination of if, switch, and for statement headers:
//
// Header   = [ InitStmt ";" ] [ Expression ] .
// Header   = [ InitStmt ] ";" [ Condition ] ";" [ PostStmt ] .  // for_stmt only
// InitStmt = SimpleStmt .
// PostStmt = SimpleStmt .
func (p *parser) header(for_stmt bool) (init, cond, post *Node) {
	if p.tok == '{' {
		return
	}

	outer := p.xnest
	p.xnest = -1

	if p.tok != ';' {
		// accept potential vardcl but complain
		// (for test/syntax/forvar.go)
		if for_stmt && p.tok == LVAR {
			yyerror("var declaration not allowed in for initializer")
			p.next()
		}
		init = p.simple_stmt(false, for_stmt)
		// If we have a range clause, we are done.
		if for_stmt && init.Op == ORANGE {
			cond = init
			init = nil

			p.xnest = outer
			return
		}
	}
	if p.got(';') {
		if for_stmt {
			if p.tok != ';' {
				cond = p.simple_stmt(false, false)
			}
			p.want(';')
			if p.tok != '{' {
				post = p.simple_stmt(false, false)
			}
		} else if p.tok != '{' {
			cond = p.simple_stmt(false, false)
		}
	} else {
		cond = init
		init = nil
	}

	p.xnest = outer
	return
}

func (p *parser) if_header() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("if_header")()
	}

	init, cond, _ := p.header(false)
	h := nod(OIF, nil, nil)
	if init != nil {
		h.Ninit.Set1(init)
	}
	h.Left = cond
	return h
}

// IfStmt = "if" [ SimpleStmt ";" ] Expression Block [ "else" ( IfStmt | Block ) ] .
func (p *parser) if_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("if_stmt")()
	}

	p.want(LIF)

	markdcl()

	stmt := p.if_header()
	if stmt.Left == nil {
		yyerror("missing condition in if statement")
	}

	stmt.Nbody.Set(p.loop_body("if clause"))

	if p.got(LELSE) {
		switch p.tok {
		case LIF:
			stmt.Rlist.Set1(p.if_stmt())
		case '{':
			cs := p.compound_stmt()
			if cs.Op == OBLOCK && cs.Ninit.Len() == 0 {
				stmt.Rlist.Set(cs.List.Slice())
			} else {
				stmt.Rlist.Set1(cs)
			}
		default:
			p.syntax_error("else must be followed by if or statement block")
			p.advance(LNAME, '}')
		}
	}

	popdcl()
	return stmt
}

// switch_stmt parses both expression and type switch statements.
//
// SwitchStmt     = ExprSwitchStmt | TypeSwitchStmt .
// ExprSwitchStmt = "switch" [ SimpleStmt ";" ] [ Expression ] "{" { ExprCaseClause } "}" .
// TypeSwitchStmt = "switch" [ SimpleStmt ";" ] TypeSwitchGuard "{" { TypeCaseClause } "}" .
func (p *parser) switch_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("switch_stmt")()
	}

	p.want(LSWITCH)
	markdcl()

	hdr := p.if_header()
	hdr.Op = OSWITCH

	tswitch := hdr.Left
	if tswitch != nil && tswitch.Op != OTYPESW {
		tswitch = nil
	}

	hdr.List.Set(p.caseblock_list(tswitch))
	popdcl()

	return hdr
}

// SelectStmt = "select" "{" { CommClause } "}" .
func (p *parser) select_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("select_stmt")()
	}

	p.want(LSELECT)
	hdr := nod(OSELECT, nil, nil)
	hdr.List.Set(p.caseblock_list(nil))
	return hdr
}

// Expression = UnaryExpr | Expression binary_op Expression .
func (p *parser) bexpr(prec OpPrec) *Node {
	// don't trace bexpr - only leads to overly nested trace output

	// prec is precedence of the prior/enclosing binary operator (if any),
	// so we only want to parse tokens of greater precedence.

	x := p.uexpr()
	for p.prec > prec {
		op, prec1 := p.op, p.prec
		p.next()
		x = nod(op, x, p.bexpr(prec1))
	}
	return x
}

func (p *parser) expr() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("expr")()
	}

	return p.bexpr(0)
}

func unparen(x *Node) *Node {
	for x.Op == OPAREN {
		x = x.Left
	}
	return x
}

// UnaryExpr = PrimaryExpr | unary_op UnaryExpr .
func (p *parser) uexpr() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("uexpr")()
	}

	var op Op
	switch p.tok {
	case '*':
		op = OIND

	case '&':
		p.next()
		// uexpr may have returned a parenthesized composite literal
		// (see comment in operand) - remove parentheses if any
		x := unparen(p.uexpr())
		if x.Op == OCOMPLIT {
			// Special case for &T{...}: turn into (*T){...}.
			x.Right = nod(OIND, x.Right, nil)
			x.Right.Implicit = true
		} else {
			x = nod(OADDR, x, nil)
		}
		return x

	case '+':
		op = OPLUS

	case '-':
		op = OMINUS

	case '!':
		op = ONOT

	case '^':
		op = OCOM

	case LCOMM:
		// receive op (<-x) or receive-only channel (<-chan E)
		p.next()

		// If the next token is LCHAN we still don't know if it is
		// a channel (<-chan int) or a receive op (<-chan int(ch)).
		// We only know once we have found the end of the uexpr.

		x := p.uexpr()

		// There are two cases:
		//
		//   <-chan...  => <-x is a channel type
		//   <-x        => <-x is a receive operation
		//
		// In the first case, <- must be re-associated with
		// the channel type parsed already:
		//
		//   <-(chan E)   =>  (<-chan E)
		//   <-(chan<-E)  =>  (<-chan (<-E))

		if x.Op == OTCHAN {
			// x is a channel type => re-associate <-
			dir := Csend
			t := x
			for ; t.Op == OTCHAN && dir == Csend; t = t.Left {
				dir = ChanDir(t.Etype)
				if dir == Crecv {
					// t is type <-chan E but <-<-chan E is not permitted
					// (report same error as for "type _ <-<-chan E")
					p.syntax_error("unexpected <-, expecting chan")
					// already progressed, no need to advance
				}
				t.Etype = EType(Crecv)
			}
			if dir == Csend {
				// channel dir is <- but channel element E is not a channel
				// (report same error as for "type _ <-chan<-E")
				p.syntax_error(fmt.Sprintf("unexpected %v, expecting chan", t))
				// already progressed, no need to advance
			}
			return x
		}

		// x is not a channel type => we have a receive op
		return nod(ORECV, x, nil)

	default:
		return p.pexpr(false)
	}

	// simple uexpr
	p.next()
	return nod(op, p.uexpr(), nil)
}

// pseudocall parses call-like statements that can be preceded by 'defer' and 'go'.
func (p *parser) pseudocall() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("pseudocall")()
	}

	x := p.pexpr(p.tok == '(') // keep_parens so we can report error below
	switch x.Op {
	case OCALL:
		return x
	case OPAREN:
		yyerror("expression in go/defer must not be parenthesized")
		// already progressed, no need to advance
	default:
		yyerror("expression in go/defer must be function call")
		// already progressed, no need to advance
	}
	return nil
}

// Operand     = Literal | OperandName | MethodExpr | "(" Expression ")" .
// Literal     = BasicLit | CompositeLit | FunctionLit .
// BasicLit    = int_lit | float_lit | imaginary_lit | rune_lit | string_lit .
// OperandName = identifier | QualifiedIdent.
func (p *parser) operand(keep_parens bool) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("operand")()
	}

	switch p.tok {
	case LLITERAL:
		x := nodlit(p.val)
		p.next()
		return x

	case LNAME:
		return p.name()

	case '(':
		p.next()
		p.xnest++
		x := p.expr() // expr_or_type
		p.xnest--
		p.want(')')

		// Optimization: Record presence of ()'s only where needed
		// for error reporting. Don't bother in other cases; it is
		// just a waste of memory and time.

		// Parentheses are not permitted on lhs of := .
		switch x.Op {
		case ONAME, ONONAME, OPACK, OTYPE, OLITERAL, OTYPESW:
			keep_parens = true
		}

		// Parentheses are not permitted around T in a composite
		// literal T{}. If the next token is a {, assume x is a
		// composite literal type T (it may not be, { could be
		// the opening brace of a block, but we don't know yet).
		if p.tok == '{' {
			keep_parens = true
		}

		// Parentheses are also not permitted around the expression
		// in a go/defer statement. In that case, operand is called
		// with keep_parens set.
		if keep_parens {
			x = nod(OPAREN, x, nil)
		}
		return x

	case LFUNC:
		t := p.ntype() // fntype
		if p.tok == '{' {
			// fnlitdcl
			closurehdr(t)
			// fnliteral
			p.next() // consume '{'
			p.fnest++
			p.xnest++
			body := p.stmt_list()
			p.xnest--
			p.fnest--
			p.want('}')
			return closurebody(body)
		}
		return t

	case '[', LCHAN, LMAP, LSTRUCT, LINTERFACE:
		return p.ntype() // othertype

	case '{':
		// common case: p.header is missing simple_stmt before { in if, for, switch
		p.syntax_error("missing operand")
		// '{' will be consumed in pexpr - no need to consume it here
		return nil

	default:
		p.syntax_error("expecting expression")
		p.advance()
		return nil
	}

	// Syntactically, composite literals are operands. Because a complit
	// type may be a qualified identifier which is handled by pexpr
	// (together with selector expressions), complits are parsed there
	// as well (operand is only called from pexpr).
}

// PrimaryExpr =
// 	Operand |
// 	Conversion |
// 	PrimaryExpr Selector |
// 	PrimaryExpr Index |
// 	PrimaryExpr Slice |
// 	PrimaryExpr TypeAssertion |
// 	PrimaryExpr Arguments .
//
// Selector       = "." identifier .
// Index          = "[" Expression "]" .
// Slice          = "[" ( [ Expression ] ":" [ Expression ] ) |
//                      ( [ Expression ] ":" Expression ":" Expression )
//                  "]" .
// TypeAssertion  = "." "(" Type ")" .
// Arguments      = "(" [ ( ExpressionList | Type [ "," ExpressionList ] ) [ "..." ] [ "," ] ] ")" .
func (p *parser) pexpr(keep_parens bool) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("pexpr")()
	}

	x := p.operand(keep_parens)

loop:
	for {
		switch p.tok {
		case '.':
			p.next()
			switch p.tok {
			case LNAME:
				// pexpr '.' sym
				x = p.new_dotname(x)

			case '(':
				p.next()
				switch p.tok {
				default:
					// pexpr '.' '(' expr_or_type ')'
					t := p.expr() // expr_or_type
					p.want(')')
					x = nod(ODOTTYPE, x, t)

				case LTYPE:
					// pexpr '.' '(' LTYPE ')'
					p.next()
					p.want(')')
					x = nod(OTYPESW, nil, x)
				}

			default:
				p.syntax_error("expecting name or (")
				p.advance(';', '}')
			}

		case '[':
			p.next()
			p.xnest++
			var index [3]*Node
			if p.tok != ':' {
				index[0] = p.expr()
			}
			ncol := 0
			for ncol < len(index)-1 && p.got(':') {
				ncol++
				if p.tok != EOF && p.tok != ':' && p.tok != ']' {
					index[ncol] = p.expr()
				}
			}
			p.xnest--
			p.want(']')

			switch ncol {
			case 0:
				i := index[0]
				if i == nil {
					yyerror("missing index in index expression")
				}
				x = nod(OINDEX, x, i)
			case 1:
				x = nod(OSLICE, x, nil)
				x.SetSliceBounds(index[0], index[1], nil)
			case 2:
				if index[1] == nil {
					yyerror("middle index required in 3-index slice")
				}
				if index[2] == nil {
					yyerror("final index required in 3-index slice")
				}
				x = nod(OSLICE3, x, nil)
				x.SetSliceBounds(index[0], index[1], index[2])

			default:
				panic("unreachable")
			}

		case '(':
			// convtype '(' expr ocomma ')'
			args, ddd := p.arg_list()

			// call or conversion
			x = nod(OCALL, x, nil)
			x.List.Set(args)
			x.Isddd = ddd

		case '{':
			// operand may have returned a parenthesized complit
			// type; accept it but complain if we have a complit
			t := unparen(x)
			// determine if '{' belongs to a complit or a compound_stmt
			complit_ok := false
			switch t.Op {
			case ONAME, ONONAME, OTYPE, OPACK, OXDOT, ODOT:
				if p.xnest >= 0 {
					// x is considered a comptype
					complit_ok = true
				}
			case OTARRAY, OTSTRUCT, OTMAP:
				// x is a comptype
				complit_ok = true
			}
			if !complit_ok {
				break loop
			}
			if t != x {
				p.syntax_error("cannot parenthesize type in composite literal")
				// already progressed, no need to advance
			}
			n := p.complitexpr()
			n.Right = x
			x = n

		default:
			break loop
		}
	}

	return x
}

// KeyedElement = [ Key ":" ] Element .
func (p *parser) keyval() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("keyval")()
	}

	// A composite literal commonly spans several lines,
	// so the line number on errors may be misleading.
	// Wrap values (but not keys!) that don't carry line
	// numbers.

	x := p.bare_complitexpr()

	if p.got(':') {
		// key ':' value
		return nod(OKEY, x, wrapname(p.bare_complitexpr()))
	}

	// value
	return wrapname(x)
}

func wrapname(x *Node) *Node {
	// These nodes do not carry line numbers.
	// Introduce a wrapper node to give the correct line.
	switch x.Op {
	case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
		x = nod(OPAREN, x, nil)
		x.Implicit = true
	}
	return x
}

// Element = Expression | LiteralValue .
func (p *parser) bare_complitexpr() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("bare_complitexpr")()
	}

	if p.tok == '{' {
		// '{' start_complit braced_keyval_list '}'
		return p.complitexpr()
	}

	return p.expr()
}

// LiteralValue = "{" [ ElementList [ "," ] ] "}" .
func (p *parser) complitexpr() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("complitexpr")()
	}

	// make node early so we get the right line number
	n := nod(OCOMPLIT, nil, nil)

	p.want('{')
	p.xnest++

	var l []*Node
	for p.tok != EOF && p.tok != '}' {
		l = append(l, p.keyval())
		if !p.ocomma('}') {
			break
		}
	}

	p.xnest--
	p.want('}')

	n.List.Set(l)
	return n
}

// names and types
//	newname is used before declared
//	oldname is used after declared
func (p *parser) new_name(sym *Sym) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("new_name")()
	}

	if sym != nil {
		return newname(sym)
	}
	return nil
}

func (p *parser) dcl_name() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("dcl_name")()
	}

	symlineno := lineno
	sym := p.sym()
	if sym == nil {
		yyerrorl(symlineno, "invalid declaration")
		return nil
	}
	return dclname(sym)
}

func (p *parser) onew_name() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("onew_name")()
	}

	if p.tok == LNAME {
		return p.new_name(p.sym())
	}
	return nil
}

func (p *parser) sym() *Sym {
	if p.tok == LNAME {
		s := p.sym_ // from localpkg
		p.next()
		return s
	}

	p.syntax_error("expecting name")
	p.advance()
	return new(Sym)
}

func mkname(sym *Sym) *Node {
	n := oldname(sym)
	if n.Name != nil && n.Name.Pack != nil {
		n.Name.Pack.Used = true
	}
	return n
}

func (p *parser) name() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("name")()
	}

	return mkname(p.sym())
}

// [ "..." ] Type
func (p *parser) dotdotdot() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("dotdotdot")()
	}

	p.want(LDDD)
	if typ := p.try_ntype(); typ != nil {
		return nod(ODDD, typ, nil)
	}

	yyerror("final argument in variadic function missing type")
	return nod(ODDD, typenod(typ(TINTER)), nil)
}

func (p *parser) ntype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("ntype")()
	}

	if typ := p.try_ntype(); typ != nil {
		return typ
	}

	p.syntax_error("")
	p.advance()
	return nil
}

// signature parses a function signature and returns an OTFUNC node.
//
// Signature = Parameters [ Result ] .
func (p *parser) signature(recv *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("signature")()
	}

	params := p.param_list(true)

	var result []*Node
	if p.tok == '(' {
		result = p.param_list(false)
	} else if t := p.try_ntype(); t != nil {
		result = []*Node{nod(ODCLFIELD, nil, t)}
	}

	typ := nod(OTFUNC, recv, nil)
	typ.List.Set(params)
	typ.Rlist.Set(result)

	return typ
}

// try_ntype is like ntype but it returns nil if there was no type
// instead of reporting an error.
//
// Type     = TypeName | TypeLit | "(" Type ")" .
// TypeName = identifier | QualifiedIdent .
// TypeLit  = ArrayType | StructType | PointerType | FunctionType | InterfaceType |
// 	      SliceType | MapType | ChannelType .
func (p *parser) try_ntype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("try_ntype")()
	}

	switch p.tok {
	case LCOMM:
		// recvchantype
		p.next()
		p.want(LCHAN)
		t := nod(OTCHAN, p.chan_elem(), nil)
		t.Etype = EType(Crecv)
		return t

	case LFUNC:
		// fntype
		p.next()
		return p.signature(nil)

	case '[':
		// '[' oexpr ']' ntype
		// '[' LDDD ']' ntype
		p.next()
		p.xnest++
		var len *Node
		if p.tok != ']' {
			if p.got(LDDD) {
				len = nod(ODDD, nil, nil)
			} else {
				len = p.expr()
			}
		}
		p.xnest--
		p.want(']')
		return nod(OTARRAY, len, p.ntype())

	case LCHAN:
		// LCHAN non_recvchantype
		// LCHAN LCOMM ntype
		p.next()
		var dir = EType(Cboth)
		if p.got(LCOMM) {
			dir = EType(Csend)
		}
		t := nod(OTCHAN, p.chan_elem(), nil)
		t.Etype = dir
		return t

	case LMAP:
		// LMAP '[' ntype ']' ntype
		p.next()
		p.want('[')
		key := p.ntype()
		p.want(']')
		val := p.ntype()
		return nod(OTMAP, key, val)

	case LSTRUCT:
		return p.structtype()

	case LINTERFACE:
		return p.interfacetype()

	case '*':
		// ptrtype
		p.next()
		return nod(OIND, p.ntype(), nil)

	case LNAME:
		return p.dotname()

	case '(':
		p.next()
		t := p.ntype()
		p.want(')')
		return t

	default:
		return nil
	}
}

func (p *parser) chan_elem() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("chan_elem")()
	}

	if typ := p.try_ntype(); typ != nil {
		return typ
	}

	p.syntax_error("missing channel element type")
	// assume element type is simply absent - don't advance
	return nil
}

func (p *parser) new_dotname(obj *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("new_dotname")()
	}

	sel := p.sym()
	if obj.Op == OPACK {
		s := restrictlookup(sel.Name, obj.Name.Pkg)
		obj.Used = true
		return oldname(s)
	}
	return nodSym(OXDOT, obj, sel)
}

func (p *parser) dotname() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("dotname")()
	}

	name := p.name()
	if p.got('.') {
		return p.new_dotname(name)
	}
	return name
}

// StructType = "struct" "{" { FieldDecl ";" } "}" .
func (p *parser) structtype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("structtype")()
	}

	p.want(LSTRUCT)
	p.want('{')
	var l []*Node
	for p.tok != EOF && p.tok != '}' {
		l = append(l, p.structdcl()...)
		if !p.osemi('}') {
			break
		}
	}
	p.want('}')

	t := nod(OTSTRUCT, nil, nil)
	t.List.Set(l)
	return t
}

// InterfaceType = "interface" "{" { MethodSpec ";" } "}" .
func (p *parser) interfacetype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("interfacetype")()
	}

	p.want(LINTERFACE)
	p.want('{')
	var l []*Node
	for p.tok != EOF && p.tok != '}' {
		l = append(l, p.interfacedcl())
		if !p.osemi('}') {
			break
		}
	}
	p.want('}')

	t := nod(OTINTER, nil, nil)
	t.List.Set(l)
	return t
}

// Function stuff.
// All in one place to show how crappy it all is.

func (p *parser) xfndcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("xfndcl")()
	}

	p.want(LFUNC)
	f := p.fndcl()
	body := p.fnbody()

	if f == nil {
		return nil
	}

	f.Nbody.Set(body)
	f.Noescape = p.pragma&Noescape != 0
	if f.Noescape && len(body) != 0 {
		yyerror("can only use //go:noescape with external func implementations")
	}
	f.Func.Pragma = p.pragma
	f.Func.Endlineno = lineno

	funcbody(f)

	return f
}

// FunctionDecl = "func" FunctionName ( Function | Signature ) .
// FunctionName = identifier .
// Function     = Signature FunctionBody .
// MethodDecl   = "func" Receiver MethodName ( Function | Signature ) .
// Receiver     = Parameters .
func (p *parser) fndcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("fndcl")()
	}

	switch p.tok {
	case LNAME:
		// FunctionName Signature
		name := p.sym()
		t := p.signature(nil)

		if name.Name == "init" {
			name = renameinit()
			if t.List.Len() > 0 || t.Rlist.Len() > 0 {
				yyerror("func init must have no arguments and no return values")
			}
		}

		if localpkg.Name == "main" && name.Name == "main" {
			if t.List.Len() > 0 || t.Rlist.Len() > 0 {
				yyerror("func main must have no arguments and no return values")
			}
		}

		f := nod(ODCLFUNC, nil, nil)
		f.Func.Nname = newfuncname(name)
		f.Func.Nname.Name.Defn = f
		f.Func.Nname.Name.Param.Ntype = t // TODO: check if nname already has an ntype
		declare(f.Func.Nname, PFUNC)

		funchdr(f)
		return f

	case '(':
		// Receiver MethodName Signature
		rparam := p.param_list(false)
		var recv *Node
		if len(rparam) > 0 {
			recv = rparam[0]
		}
		name := p.sym()
		t := p.signature(recv)

		// check after parsing header for fault-tolerance
		if recv == nil {
			yyerror("method has no receiver")
			return nil
		}

		if len(rparam) > 1 {
			yyerror("method has multiple receivers")
			return nil
		}

		if recv.Op != ODCLFIELD {
			yyerror("bad receiver in method")
			return nil
		}

		f := nod(ODCLFUNC, nil, nil)
		f.Func.Shortname = newfuncname(name)
		f.Func.Nname = methodname(f.Func.Shortname, recv.Right)
		f.Func.Nname.Name.Defn = f
		f.Func.Nname.Name.Param.Ntype = t
		declare(f.Func.Nname, PFUNC)

		funchdr(f)
		return f

	default:
		p.syntax_error("expecting name or (")
		p.advance('{', ';')
		return nil
	}
}

// FunctionBody = Block .
func (p *parser) fnbody() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("fnbody")()
	}

	if p.got('{') {
		p.fnest++
		body := p.stmt_list()
		p.fnest--
		p.want('}')
		if body == nil {
			body = []*Node{nod(OEMPTY, nil, nil)}
		}
		return body
	}

	return nil
}

// Declaration  = ConstDecl | TypeDecl | VarDecl .
// TopLevelDecl = Declaration | FunctionDecl | MethodDecl .
func (p *parser) xdcl_list() (l []*Node) {
	if trace && Debug['x'] != 0 {
		defer p.trace("xdcl_list")()
	}

	for p.tok != EOF {
		switch p.tok {
		case LVAR, LCONST, LTYPE:
			l = append(l, p.common_dcl()...)

		case LFUNC:
			l = append(l, p.xfndcl())

		default:
			if p.tok == '{' && len(l) != 0 && l[len(l)-1].Op == ODCLFUNC && l[len(l)-1].Nbody.Len() == 0 {
				// opening { of function declaration on next line
				p.syntax_error("unexpected semicolon or newline before {")
			} else {
				p.syntax_error("non-declaration statement outside function body")
			}
			p.advance(LVAR, LCONST, LTYPE, LFUNC)
			continue
		}

		// Reset p.pragma BEFORE advancing to the next token (consuming ';')
		// since comments before may set pragmas for the next function decl.
		p.pragma = 0

		if p.tok != EOF && !p.got(';') {
			p.syntax_error("after top level declaration")
			p.advance(LVAR, LCONST, LTYPE, LFUNC)
		}
	}

	if nsyntaxerrors == 0 {
		testdclstack()
	}
	return
}

// FieldDecl      = (IdentifierList Type | AnonymousField) [ Tag ] .
// AnonymousField = [ "*" ] TypeName .
// Tag            = string_lit .
func (p *parser) structdcl() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("structdcl")()
	}

	var sym *Sym
	switch p.tok {
	case LNAME:
		sym = p.sym_
		p.next()
		if sym == nil {
			panic("unreachable") // we must have a sym for LNAME
		}
		if p.tok == '.' || p.tok == LLITERAL || p.tok == ';' || p.tok == '}' {
			// embed oliteral
			field := p.embed(sym)
			tag := p.oliteral()

			field.SetVal(tag)
			return []*Node{field}
		}

		// new_name_list ntype oliteral
		fields := p.new_name_list(sym)
		typ := p.ntype()
		tag := p.oliteral()

		if len(fields) == 0 || fields[0].Sym.Name == "?" {
			// ? symbol, during import
			n := typ
			if n.Op == OIND {
				n = n.Left
			}
			n = embedded(n.Sym, importpkg)
			n.Right = typ
			n.SetVal(tag)
			return []*Node{n}
		}

		for i, n := range fields {
			fields[i] = nod(ODCLFIELD, n, typ)
			fields[i].SetVal(tag)
		}
		return fields

	case '(':
		p.next()
		if p.got('*') {
			// '(' '*' embed ')' oliteral
			field := p.embed(nil)
			p.want(')')
			tag := p.oliteral()

			field.Right = nod(OIND, field.Right, nil)
			field.SetVal(tag)
			yyerror("cannot parenthesize embedded type")
			return []*Node{field}

		} else {
			// '(' embed ')' oliteral
			field := p.embed(nil)
			p.want(')')
			tag := p.oliteral()

			field.SetVal(tag)
			yyerror("cannot parenthesize embedded type")
			return []*Node{field}
		}

	case '*':
		p.next()
		if p.got('(') {
			// '*' '(' embed ')' oliteral
			field := p.embed(nil)
			p.want(')')
			tag := p.oliteral()

			field.Right = nod(OIND, field.Right, nil)
			field.SetVal(tag)
			yyerror("cannot parenthesize embedded type")
			return []*Node{field}

		} else {
			// '*' embed oliteral
			field := p.embed(nil)
			tag := p.oliteral()

			field.Right = nod(OIND, field.Right, nil)
			field.SetVal(tag)
			return []*Node{field}
		}

	default:
		p.syntax_error("expecting field name or embedded type")
		p.advance(';', '}')
		return nil
	}
}

func (p *parser) oliteral() (v Val) {
	if p.tok == LLITERAL {
		v = p.val
		p.next()
	}
	return
}

func (p *parser) packname(name *Sym) *Sym {
	if trace && Debug['x'] != 0 {
		defer p.trace("embed")()
	}

	if name != nil {
		// LNAME was already consumed and is coming in as name
	} else if p.tok == LNAME {
		name = p.sym_
		p.next()
	} else {
		p.syntax_error("expecting name")
		p.advance('.', ';', '}')
		name = new(Sym)
	}

	if p.got('.') {
		// LNAME '.' sym
		s := p.sym()

		var pkg *Pkg
		if name.Def == nil || name.Def.Op != OPACK {
			yyerror("%v is not a package", name)
			pkg = localpkg
		} else {
			name.Def.Used = true
			pkg = name.Def.Name.Pkg
		}
		return restrictlookup(s.Name, pkg)
	}

	// LNAME
	if n := oldname(name); n.Name != nil && n.Name.Pack != nil {
		n.Name.Pack.Used = true
	}
	return name
}

func (p *parser) embed(sym *Sym) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("embed")()
	}

	pkgname := p.packname(sym)
	return embedded(pkgname, localpkg)
}

// MethodSpec        = MethodName Signature | InterfaceTypeName .
// MethodName        = identifier .
// InterfaceTypeName = TypeName .
func (p *parser) interfacedcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("interfacedcl")()
	}

	switch p.tok {
	case LNAME:
		sym := p.sym_
		p.next()

		// accept potential name list but complain
		hasNameList := false
		for p.got(',') {
			p.sym()
			hasNameList = true
		}
		if hasNameList {
			p.syntax_error("name list not allowed in interface type")
			// already progressed, no need to advance
		}

		if p.tok != '(' {
			// packname
			pname := p.packname(sym)
			return nod(ODCLFIELD, nil, oldname(pname))
		}

		// MethodName Signature
		mname := newname(sym)
		sig := p.signature(fakethis())

		meth := nod(ODCLFIELD, mname, sig)
		ifacedcl(meth)
		return meth

	case '(':
		p.next()
		pname := p.packname(nil)
		p.want(')')
		n := nod(ODCLFIELD, nil, oldname(pname))
		yyerror("cannot parenthesize embedded type")
		return n

	default:
		p.syntax_error("")
		p.advance(';', '}')
		return nil
	}
}

// param parses and returns a function parameter list entry which may be
// a parameter name and type pair (name, typ), a single type (nil, typ),
// or a single name (name, nil). In the last case, the name may still be
// a type name. The result is (nil, nil) in case of a syntax error.
//
// [ParameterName] Type
func (p *parser) param() (name *Sym, typ *Node) {
	if trace && Debug['x'] != 0 {
		defer p.trace("param")()
	}

	switch p.tok {
	case LNAME:
		name = p.sym()
		switch p.tok {
		case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', LNAME, '(':
			// sym name_or_type
			typ = p.ntype()

		case LDDD:
			// sym dotdotdot
			typ = p.dotdotdot()

		default:
			// name_or_type
			if p.got('.') {
				// a qualified name cannot be a parameter name
				typ = p.new_dotname(mkname(name))
				name = nil
			}
		}

	case LDDD:
		// dotdotdot
		typ = p.dotdotdot()

	case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', '(':
		// name_or_type
		typ = p.ntype()

	default:
		p.syntax_error("expecting )")
		p.advance(',', ')')
	}

	return
}

// Parameters    = "(" [ ParameterList [ "," ] ] ")" .
// ParameterList = ParameterDecl { "," ParameterDecl } .
// ParameterDecl = [ IdentifierList ] [ "..." ] Type .
func (p *parser) param_list(dddOk bool) []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("param_list")()
	}

	type param struct {
		name *Sym
		typ  *Node
	}
	var params []param
	var named int // number of parameters that have a name and type

	p.want('(')
	for p.tok != EOF && p.tok != ')' {
		name, typ := p.param()
		params = append(params, param{name, typ})
		if name != nil && typ != nil {
			named++
		}
		if !p.ocomma(')') {
			break
		}
	}
	p.want(')')
	// 0 <= named <= len(params)

	// There are 3 cases:
	//
	// 1) named == 0:
	//    No parameter list entry has both a name and a type; i.e. there are only
	//    unnamed parameters. Any name must be a type name; they are "converted"
	//    to types when creating the final parameter list.
	//    In case of a syntax error, there is neither a name nor a type.
	//    Nil checks take care of this.
	//
	// 2) named == len(names):
	//    All parameter list entries have both a name and a type.
	//
	// 3) Otherwise:
	if named != 0 && named != len(params) {
		// Some parameter list entries have both a name and a type:
		// Distribute types backwards and check that there are no
		// mixed named and unnamed parameters.
		var T *Node // type T in a parameter sequence: a, b, c T
		for i := len(params) - 1; i >= 0; i-- {
			p := &params[i]
			if t := p.typ; t != nil {
				// explicit type: use type for earlier parameters
				T = t
				// an explicitly typed entry must have a name
				if p.name == nil {
					T = nil // error
				}
			} else {
				// no explicit type: use type of next parameter
				p.typ = T
			}
			if T == nil {
				yyerror("mixed named and unnamed function parameters")
				break
			}
		}
		// Unless there was an error, now all parameter entries have a type.
	}

	// create final parameter list
	list := make([]*Node, len(params))
	for i, p := range params {
		// create dcl node
		var name, typ *Node
		if p.typ != nil {
			typ = p.typ
			if p.name != nil {
				// name must be a parameter name
				name = newname(p.name)
			}
		} else if p.name != nil {
			// p.name must be a type name (or nil in case of syntax error)
			typ = mkname(p.name)
		}
		n := nod(ODCLFIELD, name, typ)

		// rewrite ...T parameter
		if typ != nil && typ.Op == ODDD {
			if !dddOk {
				yyerror("cannot use ... in receiver or result parameter list")
			} else if i+1 < len(params) {
				yyerror("can only use ... with final parameter in list")
			}
			typ.Op = OTARRAY
			typ.Right = typ.Left
			typ.Left = nil
			n.Isddd = true
			if n.Left != nil {
				n.Left.Isddd = true
			}
		}

		list[i] = n
	}

	return list
}

var missing_stmt = nod(OXXX, nil, nil)

// Statement =
// 	Declaration | LabeledStmt | SimpleStmt |
// 	GoStmt | ReturnStmt | BreakStmt | ContinueStmt | GotoStmt |
// 	FallthroughStmt | Block | IfStmt | SwitchStmt | SelectStmt | ForStmt |
// 	DeferStmt .
//
// stmt may return missing_stmt.
func (p *parser) stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("stmt")()
	}

	switch p.tok {
	case '{':
		return p.compound_stmt()

	case LVAR, LCONST, LTYPE:
		return liststmt(p.common_dcl())

	case LNAME, LLITERAL, LFUNC, '(', // operands
		'[', LSTRUCT, LMAP, LCHAN, LINTERFACE, // composite types
		'+', '-', '*', '&', '^', LCOMM, '!': // unary operators
		return p.simple_stmt(true, false)

	case LFOR:
		return p.for_stmt()

	case LSWITCH:
		return p.switch_stmt()

	case LSELECT:
		return p.select_stmt()

	case LIF:
		return p.if_stmt()

	case LFALL:
		p.next()
		// will be converted to OFALL
		stmt := nod(OXFALL, nil, nil)
		stmt.Xoffset = int64(block)
		return stmt

	case LBREAK:
		p.next()
		return nod(OBREAK, p.onew_name(), nil)

	case LCONTINUE:
		p.next()
		return nod(OCONTINUE, p.onew_name(), nil)

	case LGO:
		p.next()
		return nod(OPROC, p.pseudocall(), nil)

	case LDEFER:
		p.next()
		return nod(ODEFER, p.pseudocall(), nil)

	case LGOTO:
		p.next()
		stmt := nod(OGOTO, p.new_name(p.sym()), nil)
		stmt.Sym = dclstack // context, for goto restrictions
		return stmt

	case LRETURN:
		p.next()
		var results []*Node
		if p.tok != ';' && p.tok != '}' {
			results = p.expr_list()
		}

		stmt := nod(ORETURN, nil, nil)
		stmt.List.Set(results)
		if stmt.List.Len() == 0 && Curfn != nil {
			for _, ln := range Curfn.Func.Dcl {
				if ln.Class == PPARAM {
					continue
				}
				if ln.Class != PPARAMOUT {
					break
				}
				if ln.Sym.Def != ln {
					yyerror("%s is shadowed during return", ln.Sym.Name)
				}
			}
		}

		return stmt

	case ';':
		return nil

	default:
		return missing_stmt
	}
}

// StatementList = { Statement ";" } .
func (p *parser) stmt_list() (l []*Node) {
	if trace && Debug['x'] != 0 {
		defer p.trace("stmt_list")()
	}

	for p.tok != EOF && p.tok != '}' && p.tok != LCASE && p.tok != LDEFAULT {
		s := p.stmt()
		if s == missing_stmt {
			break
		}
		if s == nil {
		} else if s.Op == OBLOCK && s.Ninit.Len() == 0 {
			l = append(l, s.List.Slice()...)
		} else {
			l = append(l, s)
		}
		// customized version of osemi:
		// ';' is optional before a closing ')' or '}'
		if p.tok == ')' || p.tok == '}' {
			continue
		}
		if !p.got(';') {
			p.syntax_error("at end of statement")
			p.advance(';', '}')
		}
	}
	return
}

// IdentifierList = identifier { "," identifier } .
//
// If first != nil we have the first symbol already.
func (p *parser) new_name_list(first *Sym) []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("new_name_list")()
	}

	if first == nil {
		first = p.sym() // may still be nil
	}
	var l []*Node
	n := p.new_name(first)
	if n != nil {
		l = append(l, n)
	}
	for p.got(',') {
		n = p.new_name(p.sym())
		if n != nil {
			l = append(l, n)
		}
	}
	return l
}

// IdentifierList = identifier { "," identifier } .
func (p *parser) dcl_name_list() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("dcl_name_list")()
	}

	s := []*Node{p.dcl_name()}
	for p.got(',') {
		s = append(s, p.dcl_name())
	}
	return s
}

// ExpressionList = Expression { "," Expression } .
func (p *parser) expr_list() []*Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("expr_list")()
	}

	l := []*Node{p.expr()}
	for p.got(',') {
		l = append(l, p.expr())
	}
	return l
}

// Arguments = "(" [ ( ExpressionList | Type [ "," ExpressionList ] ) [ "..." ] [ "," ] ] ")" .
func (p *parser) arg_list() (l []*Node, ddd bool) {
	if trace && Debug['x'] != 0 {
		defer p.trace("arg_list")()
	}

	p.want('(')
	p.xnest++

	for p.tok != EOF && p.tok != ')' && !ddd {
		l = append(l, p.expr()) // expr_or_type
		ddd = p.got(LDDD)
		if !p.ocomma(')') {
			break
		}
	}

	p.xnest--
	p.want(')')

	return
}

// osemi parses an optional semicolon.
func (p *parser) osemi(follow int32) bool {
	switch p.tok {
	case ';':
		p.next()
		return true

	case ')', '}':
		// semicolon is optional before ) or }
		return true
	}

	p.syntax_error("expecting semicolon, newline, or " + tokstring(follow))
	p.advance(follow)
	return false
}

// ocomma parses an optional comma.
func (p *parser) ocomma(follow int32) bool {
	switch p.tok {
	case ',':
		p.next()
		return true

	case ')', '}':
		// comma is optional before ) or }
		return true
	}

	p.syntax_error("expecting comma or " + tokstring(follow))
	p.advance(follow)
	return false
}
