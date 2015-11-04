// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"strconv"
	"strings"
)

const trace = true // if set, parse tracing can be enabled with -x

// TODO(gri) Once we handle imports w/o redirecting the underlying
// source of the lexer we can get rid of these. They are here for
// compatibility with the existing yacc-based parser setup.
var thenewparser parser // the parser in use
var savedstate []parser // saved parser state, used during import

func push_parser() {
	savedstate = append(savedstate, thenewparser)
	thenewparser = parser{}
	thenewparser.next()
}

func pop_parser() {
	n := len(savedstate) - 1
	thenewparser = savedstate[n]
	savedstate = savedstate[:n]
}

func parse_file() {
	// This doesn't quite work w/ the trybots. Fun experiment but we need to find a better way.
	// go func() {
	// 	prev := lexlineno
	// 	for {
	// 		time.Sleep(5 * time.Second)
	// 		t := lexlineno // racy but we don't care - any new value will do
	// 		if prev == t {
	// 			// If lexlineno doesn't change anymore we probably have an endless
	// 			// loop somewhere. Terminate before process becomes unresponsive.
	// 			Yyerror("internal error: compiler makes no progress (workaround: -oldparser)")
	// 			errorexit()
	// 		}
	// 		prev = t
	// 	}
	// }()

	thenewparser = parser{}
	thenewparser.loadsys()
	thenewparser.next()
	thenewparser.file()
}

// This loads the definitions for the low-level runtime functions,
// so that the compiler can generate calls to them,
// but does not make the name "runtime" visible as a package.
//
// go.y:loadsys
func (p *parser) loadsys() {
	if trace && Debug['x'] != 0 {
		defer p.trace("loadsys")()
	}

	importpkg = Runtimepkg

	if Debug['A'] != 0 {
		cannedimports("runtime.Builtin", "package runtime\n\n$$\n\n")
	} else {
		cannedimports("runtime.Builtin", runtimeimport)
	}
	curio.importsafe = true

	p.import_package()
	p.import_there()

	importpkg = nil
}

type parser struct {
	tok    int32     // next token (one-token look-ahead)
	op     Op        // valid if tok == LASOP
	val    Val       // valid if tok == LLITERAL
	sym_   *Sym      // valid if tok == LNAME
	nest   int       // expression nesting level (for complit ambiguity resolution)
	yy     yySymType // for temporary use by next
	indent int       // tracing support
}

func (p *parser) next() {
	p.tok = yylex(&p.yy)
	p.op = Op(p.yy.i)
	p.val = p.yy.val
	p.sym_ = p.yy.sym
}

func (p *parser) got(tok int32) bool {
	if p.tok == tok {
		p.next()
		return true
	}
	return false
}

func (p *parser) want(tok int32) {
	if p.tok != EOF && !p.got(tok) {
		p.error("")
	}
}

// ----------------------------------------------------------------------------
// Syntax error handling

// TODO(gri) Approach this more systematically. For now it passes all tests.

func syntax_error(msg string) {
	Yyerror("syntax error: " + msg)
}

func (p *parser) error(context string) {
	if p.tok == EOF {
		return
	}
	syntax_error("unexpected " + tokstring(p.tok) + context)
	// TODO(gri) keep also track of nesting below
	switch p.tok {
	case '(':
		// skip to closing ')'
		for p.tok != EOF && p.tok != ')' {
			p.next()
		}
	case '{':
		// skip to closing '}'
		for p.tok != EOF && p.tok != '}' {
			p.next()
		}
	}
	p.next() // make progress
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
	return yyTokname(int(tok))
}

// TODO(gri) figure out why yyTokname doesn't work for us as expected
var tokstrings = map[int32]string{
	LLITERAL:           "LLITERAL",
	LASOP:              "op=",
	LCOLAS:             ":=",
	LBREAK:             "break",
	LCASE:              "case",
	LCHAN:              "chan",
	LCONST:             "const",
	LCONTINUE:          "continue",
	LDDD:               "...",
	LDEFAULT:           "default",
	LDEFER:             "defer",
	LELSE:              "else",
	LFALL:              "fallthrough",
	LFOR:               "for",
	LFUNC:              "func",
	LGO:                "go",
	LGOTO:              "goto",
	LIF:                "if",
	LIMPORT:            "import",
	LINTERFACE:         "interface",
	LMAP:               "map",
	LNAME:              "<name>",
	LPACKAGE:           "package",
	LRANGE:             "range",
	LRETURN:            "return",
	LSELECT:            "select",
	LSTRUCT:            "struct",
	LSWITCH:            "switch",
	LTYPE:              "type",
	LVAR:               "var",
	LANDAND:            "&&",
	LANDNOT:            "&^",
	LBODY:              "LBODY", // we should never see this one
	LCOMM:              "<-",
	LDEC:               "--",
	LEQ:                "==",
	LGE:                ">=",
	LGT:                ">",
	LIGNORE:            "LIGNORE", // we should never see this one
	LINC:               "++",
	LLE:                "<=",
	LLSH:               "<<",
	LLT:                "<",
	LNE:                "!=",
	LOROR:              "||",
	LRSH:               ">>",
	NotPackage:         "NotPackage",         // we should never see this one
	NotParen:           "NotParen",           // we should never see this one
	PreferToRightParen: "PreferToRightParen", // we should never see this one
}

func (p *parser) print_trace(msg ...interface{}) {
	const dots = ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . "
	const n = len(dots)
	fmt.Printf("%5d: ", lineno)

	// TODO(gri) imports screw up p.indent - fix this
	if p.indent < 0 {
		p.indent = 0
	}

	i := 2 * p.indent
	for i > n {
		fmt.Print(dots)
		i -= n
	}
	// i <= n
	fmt.Print(dots[0:i])
	fmt.Println(msg...)
}

// usage: defer p.trace(msg)()
func (p *parser) trace(msg string) func() {
	p.print_trace(msg, "(")
	p.indent++
	return func() {
		p.indent--
		if x := recover(); x != nil {
			panic(x) // skip print_trace
		}
		p.print_trace(")")
	}
}

// ----------------------------------------------------------------------------
// Parsing package files

// go.y:file
func (p *parser) file() {
	if trace && Debug['x'] != 0 {
		defer p.trace("file")()
	}

	p.package_()

	//go.y:imports
	for p.tok == LIMPORT {
		p.import_()
		p.want(';')
	}

	xtop = concat(xtop, p.xdcl_list())
}

// go.y:package
func (p *parser) package_() {
	if trace && Debug['x'] != 0 {
		defer p.trace("package_")()
	}

	if p.got(LPACKAGE) {
		mkpackage(p.sym().Name)
		p.want(';')
	} else {
		prevlineno = lineno // TODO(gri) do we still need this? (e.g., not needed for test/fixedbugs/bug050.go)
		Yyerror("package statement must be first")
		errorexit()
	}
}

// import:
// 	LIMPORT import_stmt
// |	LIMPORT '(' import_stmt_list osemi ')'
// |	LIMPORT '(' ')'

func (p *parser) import_() {
	if trace && Debug['x'] != 0 {
		defer p.trace("import_")()
	}

	p.want(LIMPORT)
	if p.got('(') {
		for p.tok != EOF && p.tok != ')' {
			p.import_stmt()
			p.osemi()
		}
		p.want(')')
	} else {
		p.import_stmt()
	}
}

// go.y:import_stmt
func (p *parser) import_stmt() {
	if trace && Debug['x'] != 0 {
		defer p.trace("import_stmt")()
	}

	line := int32(p.import_here())
	if p.tok == LPACKAGE {
		p.import_package()
		p.import_there()

		ipkg := importpkg
		my := importmyname
		importpkg = nil
		importmyname = nil

		if my == nil {
			my = Lookup(ipkg.Name)
		}

		pack := Nod(OPACK, nil, nil)
		pack.Sym = my
		pack.Name.Pkg = ipkg
		pack.Lineno = line

		if strings.HasPrefix(my.Name, ".") {
			importdot(ipkg, pack)
			return
		}
		if my.Name == "init" {
			lineno = line
			Yyerror("cannot import package as init - init must be a func")
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

		return
	}

	p.import_there()
	// When an invalid import path is passed to importfile,
	// it calls Yyerror and then sets up a fake import with
	// no package statement. This allows us to test more
	// than one invalid import statement in a single file.
	if nerrors == 0 {
		Fatalf("phase error in import")
	}
}

// go.y:import_here
func (p *parser) import_here() int {
	if trace && Debug['x'] != 0 {
		defer p.trace("import_here")()
	}

	importmyname = nil
	switch p.tok {
	case LNAME, '@', '?':
		// import with given name
		importmyname = p.sym()

	case '.':
		// import into my name space
		importmyname = Lookup(".")
		p.next()
	}

	var path Val
	if p.tok == LLITERAL {
		path = p.val
		p.next()
	} else {
		syntax_error("missing import path; require quoted string")
	}

	line := parserline() // TODO(gri) check correct placement of this
	importfile(&path, line)
	return line
}

// go.y:import_package
func (p *parser) import_package() {
	if trace && Debug['x'] != 0 {
		defer p.trace("import_package")()
	}

	p.want(LPACKAGE)
	var name string
	if p.tok == LNAME {
		name = p.sym_.Name
		p.next()
	} else {
		p.import_error()
	}

	// go.y:import_safety
	if p.tok == LNAME {
		if p.sym_.Name == "safe" {
			curio.importsafe = true
		}
		p.next()
	}
	p.want(';')

	if importpkg.Name == "" {
		importpkg.Name = name
		numImport[name]++
	} else if importpkg.Name != name {
		Yyerror("conflicting names %s and %s for package %q", importpkg.Name, name, importpkg.Path)
	}
	if incannedimport == 0 {
		importpkg.Direct = true
	}
	importpkg.Safe = curio.importsafe

	if safemode != 0 && !curio.importsafe {
		Yyerror("cannot import unsafe package %q", importpkg.Path)
	}
}

// go.y:import_there
func (p *parser) import_there() {
	if trace && Debug['x'] != 0 {
		defer p.trace("import_there")()
	}

	defercheckwidth()

	p.hidden_import_list()
	p.want('$')
	// don't read past 2nd '$'
	if p.tok != '$' {
		p.import_error()
	}

	resumecheckwidth()
	unimportfile()
}

// go.y:common_dcl
func (p *parser) common_dcl() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("common_dcl")()
	}

	var dcl func() *NodeList
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
	var l *NodeList
	if p.got('(') {
		for p.tok != EOF && p.tok != ')' {
			l = concat(l, dcl())
			p.osemi()
		}
		p.want(')')
	} else {
		l = dcl()
	}

	iota_ = -100000
	lastconst = nil

	return l
}

// go.y:vardcl
func (p *parser) vardcl() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("vardcl")()
	}

	names := p.dcl_name_list()
	var typ *Node
	var exprs *NodeList
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

// go.y:constdcl
func (p *parser) constdcl() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("constdcl")()
	}

	names := p.dcl_name_list()
	var typ *Node
	var exprs *NodeList
	if p.tok != EOF && p.tok != ';' && p.tok != ')' {
		if p.tok != '=' {
			typ = p.ntype()
		}
		if p.got('=') {
			exprs = p.expr_list()
		}
	}

	return constiter(names, typ, exprs)
}

// go.y:typedcl
func (p *parser) typedcl() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("typedcl")()
	}

	name := typedcl0(p.sym())

	// handle case where type is missing
	var typ *Node
	if p.tok != ';' {
		typ = p.ntype()
	} else {
		p.error(" in type declaration")
	}

	return list1(typedcl1(name, typ, true))
}

// go.y:simple_stmt
// may return missing_stmt if labelOk is set
func (p *parser) simple_stmt(labelOk, rangeOk bool) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("simple_stmt")()
	}

	if rangeOk && p.got(LRANGE) {
		// LRANGE expr
		r := Nod(ORANGE, nil, p.expr())
		r.Etype = 0 // := flag
		return r
	}

	lhs := p.expr_list()

	if count(lhs) == 1 && p.tok != '=' && p.tok != LCOLAS && p.tok != LRANGE {
		// expr
		lhs := lhs.N
		switch p.tok {
		case LASOP:
			// expr LASOP expr
			op := p.op
			p.next()
			rhs := p.expr()

			stmt := Nod(OASOP, lhs, rhs)
			stmt.Etype = EType(op) // rathole to pass opcode
			return stmt

		case LINC:
			// expr LINC
			p.next()

			stmt := Nod(OASOP, lhs, Nodintconst(1))
			stmt.Implicit = true
			stmt.Etype = EType(OADD)
			return stmt

		case LDEC:
			// expr LDEC
			p.next()

			stmt := Nod(OASOP, lhs, Nodintconst(1))
			stmt.Implicit = true
			stmt.Etype = EType(OSUB)
			return stmt

		case ':':
			// labelname ':' stmt
			if labelOk {
				// If we have a labelname, it was parsed by operand
				// (calling p.name()) and given an ONAME, ONONAME, or OTYPE node.
				if lhs.Op == ONAME || lhs.Op == ONONAME || lhs.Op == OTYPE {
					lhs = newname(lhs.Sym)
				} else {
					p.error(", expecting semicolon or newline or }")
				}
				lhs := Nod(OLABEL, lhs, nil)
				lhs.Sym = dclstack // context, for goto restrictions
				p.next()           // consume ':' after making label node for correct lineno
				return p.labeled_stmt(lhs)
			}
			fallthrough

		default:
			// expr
			// These nodes do not carry line numbers.
			// Since a bare name used as an expression is an error,
			// introduce a wrapper node to give the correct line.
			switch lhs.Op {
			case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
				lhs = Nod(OPAREN, lhs, nil)
				lhs.Implicit = true
			}
			return lhs
		}
	}

	// expr_list
	switch p.tok {
	case '=':
		p.next()
		if rangeOk && p.got(LRANGE) {
			// expr_list '=' LRANGE expr
			r := Nod(ORANGE, nil, p.expr())
			r.List = lhs
			r.Etype = 0 // := flag
			return r
		}

		// expr_list '=' expr_list
		rhs := p.expr_list()

		if lhs.Next == nil && rhs.Next == nil {
			// simple
			return Nod(OAS, lhs.N, rhs.N)
		}
		// multiple
		stmt := Nod(OAS2, nil, nil)
		stmt.List = lhs
		stmt.Rlist = rhs
		return stmt

	case LCOLAS:
		line := lineno
		p.next()

		if rangeOk && p.got(LRANGE) {
			// expr_list LCOLAS LRANGE expr
			r := Nod(ORANGE, nil, p.expr())
			r.List = lhs
			r.Colas = true
			colasdefn(lhs, r)
			return r
		}

		// expr_list LCOLAS expr_list
		rhs := p.expr_list()

		if rhs.N.Op == OTYPESW {
			ss := Nod(OTYPESW, nil, rhs.N.Right)
			if rhs.Next != nil {
				Yyerror("expr.(type) must be alone in list")
			}
			if lhs.Next != nil {
				Yyerror("argument count mismatch: %d = %d", count(lhs), 1)
			} else if (lhs.N.Op != ONAME && lhs.N.Op != OTYPE && lhs.N.Op != ONONAME && (lhs.N.Op != OLITERAL || lhs.N.Name == nil)) || isblank(lhs.N) {
				Yyerror("invalid variable name %s in type switch", lhs.N)
			} else {
				ss.Left = dclname(lhs.N.Sym)
			} // it's a colas, so must not re-use an oldname.
			return ss
		}
		return colas(lhs, rhs, int32(line))

	default:
		p.error(", expecting := or = or comma")
		return nil
	}
}

// may return missing_stmt
func (p *parser) labeled_stmt(label *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("labeled_stmt")()
	}

	var ls *Node // labeled statement
	if p.tok != '}' && p.tok != EOF {
		ls = p.stmt()
		if ls == missing_stmt {
			// report error at line of ':' token
			saved := lexlineno
			lexlineno = prevlineno
			syntax_error("missing statement after label")
			lexlineno = saved
			return missing_stmt
		}
	}

	label.Name.Defn = ls
	l := list1(label)
	if ls != nil {
		l = list(l, ls)
	}
	return liststmt(l)
}

// go.y:case
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
			markdcl()
			stmt := Nod(OXCASE, nil, nil)
			stmt.List = cases
			if tswitch != nil {
				if n := tswitch.Left; n != nil {
					// type switch - declare variable
					nn := newname(n.Sym)
					declare(nn, dclcontext)
					stmt.Rlist = list1(nn)

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
			markdcl()
			stmt := Nod(OXCASE, nil, nil)
			var n *Node
			if cases.Next == nil {
				n = Nod(OAS, cases.N, rhs)
			} else {
				n = Nod(OAS2, nil, nil)
				n.List = cases
				n.Rlist = list1(rhs)
			}
			stmt.List = list1(n)

			p.want(':') // consume ':' after declaring select cases for correct lineno
			return stmt

		case LCOLAS:
			// LCASE expr_or_type_list LCOLAS expr ':'
			p.next()
			rhs := p.expr()

			// will be converted to OCASE
			// right will point to next case
			// done in casebody()
			markdcl()
			stmt := Nod(OXCASE, nil, nil)
			stmt.List = list1(colas(cases, list1(rhs), int32(p.op)))

			p.want(':') // consume ':' after declaring select cases for correct lineno
			return stmt

		default:
			p.error(", expecting := or = or : or comma")
			return nil
		}

	case LDEFAULT:
		// LDEFAULT ':'
		p.next()

		markdcl()
		stmt := Nod(OXCASE, nil, nil)
		if tswitch != nil {
			if n := tswitch.Left; n != nil {
				// type switch - declare variable
				nn := newname(n.Sym)
				declare(nn, dclcontext)
				stmt.Rlist = list1(nn)

				// keep track of the instances for reporting unused
				nn.Name.Defn = tswitch
			}
		}

		p.want(':') // consume ':' after declaring type switch var for correct lineno
		return stmt

	default:
		p.error(", expecting case or default or }")
		return nil
	}
}

// go.y:compound_stmt
func (p *parser) compound_stmt(else_clause bool) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("compound_stmt")()
	}

	if p.tok == '{' {
		markdcl()
		p.next() // consume ';' after markdcl() for correct lineno
	} else if else_clause {
		syntax_error("else must be followed by if or statement block")
		// skip through closing }
		for p.tok != EOF && p.tok != '}' {
			p.next()
		}
		p.next()
		return nil
	} else {
		panic("unreachable")
	}

	l := p.stmt_list()

	var stmt *Node
	if l == nil {
		stmt = Nod(OEMPTY, nil, nil)
	} else {
		stmt = liststmt(l)
	}
	popdcl()

	p.want('}') // TODO(gri) is this correct location w/ respect to popdcl()?

	return stmt
}

// go.y:caseblock
func (p *parser) caseblock(tswitch *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("caseblock")()
	}

	stmt := p.case_(tswitch)

	// If the last token read by the lexer was consumed
	// as part of the case, clear it (parser has cleared yychar).
	// If the last token read by the lexer was the lookahead
	// leave it alone (parser has it cached in yychar).
	// This is so that the stmt_list action doesn't look at
	// the case tokens if the stmt_list is empty.
	//yylast = yychar;
	stmt.Xoffset = int64(block)

	stmt.Nbody = p.stmt_list()

	// TODO(gri) what do we need to do here?
	// // This is the only place in the language where a statement
	// // list is not allowed to drop the final semicolon, because
	// // it's the only place where a statement list is not followed
	// // by a closing brace.  Handle the error for pedantry.

	// // Find the final token of the statement list.
	// // yylast is lookahead; yyprev is last of stmt_list
	// last := yyprev;

	// if last > 0 && last != ';' && yychar != '}' {
	// 	Yyerror("missing statement after label");
	// }

	popdcl()

	return stmt
}

// go.y:caseblock_list
func (p *parser) caseblock_list(tswitch *Node) (l *NodeList) {
	if trace && Debug['x'] != 0 {
		defer p.trace("caseblock_list")()
	}

	if !p.got('{') {
		syntax_error("missing { after switch clause")
		// skip through closing }
		for p.tok != EOF && p.tok != '}' {
			p.next()
		}
		p.next()
		return nil
	}

	for p.tok != '}' {
		l = list(l, p.caseblock(tswitch))
	}
	p.want('}')
	return
}

// go.y:loop_body
func (p *parser) loop_body(context string) *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("loop_body")()
	}

	if p.tok == '{' {
		markdcl()
		p.next() // consume ';' after markdcl() for correct lineno
	} else {
		syntax_error("missing { after " + context)
		// skip through closing }
		for p.tok != EOF && p.tok != '}' {
			p.next()
		}
		p.next()
		return nil
	}

	body := p.stmt_list()
	popdcl()
	p.want('}')

	return body
}

// go.y:for_header
func (p *parser) for_header() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("for_header")()
	}

	init, cond, post := p.header(true)

	if init != nil || post != nil {
		// init ; test ; incr
		if post != nil && post.Colas {
			Yyerror("cannot declare in the for-increment")
		}
		h := Nod(OFOR, nil, nil)
		if init != nil {
			h.Ninit = list1(init)
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
	h := Nod(OFOR, nil, nil)
	h.Left = cond
	return h
}

// go.y:for_body
func (p *parser) for_body() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("for_body")()
	}

	stmt := p.for_header()
	body := p.loop_body("for clause")

	stmt.Nbody = concat(stmt.Nbody, body)
	return stmt
}

// go.y:for_stmt
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

func (p *parser) header(for_stmt bool) (init, cond, post *Node) {
	if p.tok == '{' {
		return
	}

	nest := p.nest
	p.nest = -1

	if p.tok != ';' {
		// accept potential vardcl but complain
		// (for test/syntax/forvar.go)
		if for_stmt && p.tok == LVAR {
			Yyerror("var declaration not allowed in for initializer")
			p.next()
		}
		init = p.simple_stmt(false, for_stmt)
		// If we have a range clause, we are done.
		if for_stmt && init.Op == ORANGE {
			cond = init
			init = nil

			p.nest = nest
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

	p.nest = nest

	return
}

// go.y:if_header
func (p *parser) if_header() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("if_header")()
	}

	init, cond, _ := p.header(false)
	h := Nod(OIF, nil, nil)
	h.Ninit = list1(init)
	h.Left = cond
	return h
}

// go.y:if_stmt
func (p *parser) if_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("if_stmt")()
	}

	p.want(LIF)

	markdcl()

	stmt := p.if_header()
	if stmt.Left == nil {
		Yyerror("missing condition in if statement")
	}

	stmt.Nbody = p.loop_body("if clause")

	l := p.elseif_list_else()

	n := stmt
	popdcl()
	for nn := l; nn != nil; nn = nn.Next {
		if nn.N.Op == OIF {
			popdcl()
		}
		n.Rlist = list1(nn.N)
		n = nn.N
	}

	return stmt
}

// go.y:elsif
func (p *parser) elseif() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("elseif")()
	}

	// LELSE LIF already consumed
	markdcl()

	stmt := p.if_header()
	if stmt.Left == nil {
		Yyerror("missing condition in if statement")
	}

	stmt.Nbody = p.loop_body("if clause")

	return list1(stmt)
}

// go.y:elsif_list
// go.y:else
func (p *parser) elseif_list_else() (l *NodeList) {
	if trace && Debug['x'] != 0 {
		defer p.trace("elseif_list_else")()
	}

	for p.got(LELSE) {
		if p.got(LIF) {
			l = concat(l, p.elseif())
		} else {
			l = concat(l, p.else_())
			break
		}
	}

	return l
}

// go.y:else
func (p *parser) else_() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("else")()
	}

	l := &NodeList{N: p.compound_stmt(true)}
	l.End = l
	return l

}

// go.y:switch_stmt
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

	hdr.List = p.caseblock_list(tswitch)
	popdcl()

	return hdr
}

// go.y:select_stmt
func (p *parser) select_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("select_stmt")()
	}

	p.want(LSELECT)
	hdr := Nod(OSELECT, nil, nil)
	hdr.List = p.caseblock_list(nil)
	return hdr
}

// TODO(gri) should have lexer return this info - no need for separate lookup
var prectab = map[int32]struct {
	prec int // > 0 (0 indicates not found)
	op   Op
}{
	// not an expression anymore, but left in so we can give a good error
	// message when used in expression context
	LCOMM: {1, OSEND},

	LOROR: {2, OOROR},

	LANDAND: {3, OANDAND},

	LEQ: {4, OEQ},
	LNE: {4, ONE},
	LLE: {4, OLE},
	LGE: {4, OGE},
	LLT: {4, OLT},
	LGT: {4, OGT},

	'+': {5, OADD},
	'-': {5, OSUB},
	'|': {5, OOR},
	'^': {5, OXOR},

	'*':     {6, OMUL},
	'/':     {6, ODIV},
	'%':     {6, OMOD},
	'&':     {6, OAND},
	LLSH:    {6, OLSH},
	LRSH:    {6, ORSH},
	LANDNOT: {6, OANDNOT},
}

func (p *parser) bexpr(prec int) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("expr")()
	}

	x := p.uexpr()
	t := prectab[p.tok]
	for tprec := t.prec; tprec >= prec; tprec-- {
		for tprec == prec {
			p.next()
			y := p.bexpr(t.prec + 1)
			x = Nod(t.op, x, y)
			t = prectab[p.tok]
			tprec = t.prec
		}
	}
	return x
}

// go.y:expr
func (p *parser) expr() *Node {
	return p.bexpr(1)
}

// go.y:uexpr
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
		x := p.uexpr()
		if x.Op == OCOMPLIT {
			// Special case for &T{...}: turn into (*T){...}.
			x.Right = Nod(OIND, x.Right, nil)
			x.Right.Implicit = true
		} else {
			x = Nod(OADDR, x, nil)
		}
		return x

	case '+':
		op = OPLUS

	case '-':
		op = OMINUS

	case '!':
		op = ONOT

	case '~':
		// TODO(gri) do this in the lexer instead
		p.next()
		x := p.uexpr()
		Yyerror("the bitwise complement operator is ^")
		return Nod(OCOM, x, nil)

	case '^':
		op = OCOM

	case LCOMM:
		// receive operation (<-s2) or receive-only channel type (<-chan s3)
		p.next()
		if p.got(LCHAN) {
			// <-chan T
			t := Nod(OTCHAN, p.chan_elem(), nil)
			t.Etype = Crecv
			return t
		}
		return Nod(ORECV, p.uexpr(), nil)

	default:
		return p.pexpr(false)
	}

	// simple uexpr
	p.next()
	return Nod(op, p.uexpr(), nil)
}

// call-like statements that can be preceded by 'defer' and 'go'
//
// go.y:pseudocall
func (p *parser) pseudocall() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("pseudocall")()
	}

	x := p.pexpr(true)
	if x.Op != OCALL {
		Yyerror("argument to go/defer must be function call")
	}
	return x
}

// go.y:pexpr (partial)
func (p *parser) operand(keep_parens bool) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("operand")()
	}

	switch p.tok {
	case LLITERAL:
		x := nodlit(p.val)
		p.next()
		return x

	case LNAME, '@', '?':
		return p.name()

	case '(':
		p.next()
		p.nest++
		x := p.expr() // expr_or_type
		p.nest--
		p.want(')')

		// Need to know on lhs of := whether there are ( ).
		// Don't bother with the OPAREN in other cases:
		// it's just a waste of memory and time.
		//
		// But if the next token is a { , introduce OPAREN since
		// we may have a composite literal and we need to know
		// if there were ()'s'.
		//
		// TODO(gri) could simplify this if we parse complits
		// in operand (see respective comment in pexpr).
		if keep_parens || p.tok == '{' {
			return Nod(OPAREN, x, nil)
		}
		switch x.Op {
		case ONAME, ONONAME, OPACK, OTYPE, OLITERAL, OTYPESW:
			return Nod(OPAREN, x, nil)
		}
		return x

	case LFUNC:
		t := p.fntype()
		if p.tok == '{' {
			// fnlitdcl
			closurehdr(t)
			// fnliteral
			p.next() // consume '{'
			p.nest++
			body := p.stmt_list()
			p.nest--
			p.want('}')
			return closurebody(body)
		}
		return t

	case '[', LCHAN, LMAP, LSTRUCT, LINTERFACE:
		return p.othertype()

	case '{':
		// common case: p.header is missing simple_stmt before { in if, for, switch
		syntax_error("missing operand")
		// '{' will be consumed in pexpr - no need to consume it here
		return nil

	default:
		p.error(" in operand")
		return nil
	}
}

// go.y:pexpr, pexpr_no_paren
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
			case LNAME, '@', '?':
				// pexpr '.' sym
				sel := p.sym()

				if x.Op == OPACK {
					s := restrictlookup(sel.Name, x.Name.Pkg)
					x.Used = true
					x = oldname(s)
					break
				}
				x = Nod(OXDOT, x, newname(sel))

			case '(':
				p.next()
				switch p.tok {
				default:
					// pexpr '.' '(' expr_or_type ')'
					t := p.expr() // expr_or_type
					p.want(')')
					x = Nod(ODOTTYPE, x, t)

				case LTYPE:
					// pexpr '.' '(' LTYPE ')'
					p.next()
					p.want(')')
					x = Nod(OTYPESW, nil, x)
				}

			default:
				p.error(", expecting name or (")
			}

		case '[':
			p.next()
			p.nest++
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
			p.nest--
			p.want(']')

			switch ncol {
			case 0:
				i := index[0]
				if i == nil {
					Yyerror("missing index in index expression")
				}
				x = Nod(OINDEX, x, i)
			case 1:
				i := index[0]
				j := index[1]
				x = Nod(OSLICE, x, Nod(OKEY, i, j))
			case 2:
				i := index[0]
				j := index[1]
				k := index[2]
				if j == nil {
					Yyerror("middle index required in 3-index slice")
				}
				if k == nil {
					Yyerror("final index required in 3-index slice")
				}
				x = Nod(OSLICE3, x, Nod(OKEY, i, Nod(OKEY, j, k)))

			default:
				panic("unreachable")
			}

		case '(':
			// convtype '(' expr ocomma ')'
			p.next()
			p.nest++
			args, ddd := p.arg_list()
			p.nest--
			p.want(')')

			// call or conversion
			x = Nod(OCALL, x, nil)
			x.List = args
			x.Isddd = ddd

		case '{':
			// TODO(gri) should this (complit acceptance) be in operand?
			// accept ()'s around the complit type but complain if we have a complit
			t := x
			for t.Op == OPAREN {
				t = t.Left
			}
			// determine if '{' belongs to a complit or a compound_stmt
			complit_ok := false
			switch t.Op {
			case ONAME, ONONAME, OTYPE, OPACK, OXDOT, ODOT:
				if p.nest >= 0 {
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
				syntax_error("cannot parenthesize type in composite literal")
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

// go.y:keyval
func (p *parser) keyval() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("keyval")()
	}

	x := p.bare_complitexpr()
	if p.got(':') {
		x = Nod(OKEY, x, p.bare_complitexpr())
	}
	return x
}

// go.y:bare_complitexpr
func (p *parser) bare_complitexpr() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("bare_complitexpr")()
	}

	if p.tok == '{' {
		// '{' start_complit braced_keyval_list '}'
		return p.complitexpr()
	}

	x := p.expr()

	// These nodes do not carry line numbers.
	// Since a composite literal commonly spans several lines,
	// the line number on errors may be misleading.
	// Introduce a wrapper node to give the correct line.

	// TODO(gri) This is causing trouble when used for keys. Need to fix complit parsing.
	// switch x.Op {
	// case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
	// 	x = Nod(OPAREN, x, nil)
	// 	x.Implicit = true
	// }
	return x
}

// go.y:complitexpr
func (p *parser) complitexpr() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("complitexpr")()
	}

	// make node early so we get the right line number
	n := Nod(OCOMPLIT, nil, nil)

	p.want('{')
	p.nest++

	var l *NodeList
	for p.tok != EOF && p.tok != '}' {
		l = list(l, p.keyval())
		p.ocomma("composite literal")
	}

	p.nest--
	p.want('}')

	n.List = l
	return n
}

// names and types
//	newname is used before declared
//	oldname is used after declared
//
// go.y:new_name:
func (p *parser) new_name(sym *Sym) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("new_name")()
	}

	if sym != nil {
		return newname(sym)
	}
	return nil
}

// go.y:onew_name:
func (p *parser) onew_name() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("onew_name")()
	}

	switch p.tok {
	case LNAME, '@', '?':
		return p.new_name(p.sym())
	}
	return nil
}

// go.y:sym
func (p *parser) sym() *Sym {
	switch p.tok {
	case LNAME:
		s := p.sym_
		p.next()
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if importpkg != nil && !exportname(s.Name) {
			s = Pkglookup(s.Name, builtinpkg)
		}
		return s

	case '@':
		return p.hidden_importsym()

	case '?':
		p.next()
		return nil

	default:
		p.error("")
		return new(Sym)
	}
}

func mkname(sym *Sym) *Node {
	n := oldname(sym)
	if n.Name != nil && n.Name.Pack != nil {
		n.Name.Pack.Used = true
	}
	return n
}

// go.y:name
func (p *parser) name() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("name")()
	}

	return mkname(p.sym())
}

// go.y:dotdotdot
func (p *parser) dotdotdot() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("dotdotdot")()
	}

	p.want(LDDD)
	switch p.tok {
	case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', LNAME, '@', '?', '(':
		return Nod(ODDD, p.ntype(), nil)
	}

	Yyerror("final argument in variadic function missing type")
	return Nod(ODDD, typenod(typ(TINTER)), nil)
}

// go.y:ntype
func (p *parser) ntype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("ntype")()
	}

	switch p.tok {
	case LCOMM:
		return p.recvchantype()

	case LFUNC:
		return p.fntype()

	case '[', LCHAN, LMAP, LSTRUCT, LINTERFACE:
		return p.othertype()

	case '*':
		return p.ptrtype()

	case LNAME, '@', '?':
		return p.dotname()

	case '(':
		p.next()
		t := p.ntype()
		p.want(')')
		return t

	case LDDD:
		// permit ...T but complain
		// TODO(gri) introduced for test/fixedbugs/bug228.go - maybe adjust bug or find better solution
		p.error(" in type")
		return p.ntype()

	default:
		p.error(" in type")
		return nil
	}
}

func (p *parser) chan_elem() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("chan_elem")()
	}

	switch p.tok {
	case LCOMM, LFUNC,
		'[', LCHAN, LMAP, LSTRUCT, LINTERFACE,
		'*',
		LNAME, '@', '?',
		'(',
		LDDD:
		return p.ntype()
	default:
		syntax_error("missing channel element type")
		return nil
	}
}

// go.y:fnret_type
func (p *parser) fnret_type() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("fnret_type")()
	}

	switch p.tok {
	case LCOMM:
		return p.recvchantype()

	case LFUNC:
		return p.fntype()

	case '[', LCHAN, LMAP, LSTRUCT, LINTERFACE:
		return p.othertype()

	case '*':
		return p.ptrtype()

	default:
		return p.dotname()
	}
}

// go.y:dotname
func (p *parser) dotname() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("dotname")()
	}

	s1 := p.name()

	switch p.tok {
	default:
		return s1

	case '.':
		p.next()
		s3 := p.sym()

		if s1.Op == OPACK {
			var s *Sym
			s = restrictlookup(s3.Name, s1.Name.Pkg)
			s1.Used = true
			return oldname(s)
		}
		return Nod(OXDOT, s1, newname(s3))
	}
}

// go.y:othertype
func (p *parser) othertype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("othertype")()
	}

	switch p.tok {
	case '[':
		// '[' oexpr ']' ntype
		// '[' LDDD ']' ntype
		p.next()
		p.nest++
		var len *Node
		if p.tok != ']' {
			if p.got(LDDD) {
				len = Nod(ODDD, nil, nil)
			} else {
				len = p.expr()
			}
		}
		p.nest--
		p.want(']')
		return Nod(OTARRAY, len, p.ntype())

	case LCHAN:
		// LCHAN non_recvchantype
		// LCHAN LCOMM ntype
		p.next()
		var dir EType = Cboth
		if p.got(LCOMM) {
			dir = Csend
		}
		t := Nod(OTCHAN, p.chan_elem(), nil)
		t.Etype = dir
		return t

	case LMAP:
		// LMAP '[' ntype ']' ntype
		p.next()
		p.want('[')
		key := p.ntype()
		p.want(']')
		val := p.ntype()
		return Nod(OTMAP, key, val)

	case LSTRUCT:
		// structtype
		return p.structtype()

	case LINTERFACE:
		// interfacetype
		return p.interfacetype()

	default:
		panic("unreachable")
	}
}

// go.y:ptrtype
func (p *parser) ptrtype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("ptrtype")()
	}

	p.want('*')
	return Nod(OIND, p.ntype(), nil)
}

// go.y:recvchantype
func (p *parser) recvchantype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("recvchantype")()
	}

	p.want(LCOMM)
	p.want(LCHAN)
	t := Nod(OTCHAN, p.chan_elem(), nil)
	t.Etype = Crecv
	return t
}

// go.y:structtype
func (p *parser) structtype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("structtype")()
	}

	p.want(LSTRUCT)
	p.want('{')
	var l *NodeList
	for p.tok != EOF && p.tok != '}' {
		l = concat(l, p.structdcl())
		p.osemi()
	}
	p.want('}')

	t := Nod(OTSTRUCT, nil, nil)
	t.List = l
	return t
}

// go.y:interfacetype
func (p *parser) interfacetype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("interfacetype")()
	}

	p.want(LINTERFACE)
	p.want('{')
	var l *NodeList
	for p.tok != EOF && p.tok != '}' {
		l = list(l, p.interfacedcl())
		p.osemi()
	}
	p.want('}')

	t := Nod(OTINTER, nil, nil)
	t.List = l
	return t
}

// Function stuff.
// All in one place to show how crappy it all is.
//
// go.y:xfndcl
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
	if noescape && body != nil {
		Yyerror("can only use //go:noescape with external func implementations")
	}

	f.Nbody = body
	f.Func.Endlineno = lineno
	f.Noescape = noescape
	f.Func.Norace = norace
	f.Func.Nosplit = nosplit
	f.Func.Noinline = noinline
	f.Func.Nowritebarrier = nowritebarrier
	f.Func.Nowritebarrierrec = nowritebarrierrec
	f.Func.Systemstack = systemstack
	funcbody(f)

	return f
}

// go.y:fndcl
func (p *parser) fndcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("fndcl")()
	}

	switch p.tok {
	case LNAME, '@', '?':
		// sym '(' oarg_type_list_ocomma ')' fnres
		name := p.sym()
		params := p.param_list()
		result := p.fnres()

		params = checkarglist(params, 1)

		if name.Name == "init" {
			name = renameinit()
			if params != nil || result != nil {
				Yyerror("func init must have no arguments and no return values")
			}
		}

		if localpkg.Name == "main" && name.Name == "main" {
			if params != nil || result != nil {
				Yyerror("func main must have no arguments and no return values")
			}
		}

		t := Nod(OTFUNC, nil, nil)
		t.List = params
		t.Rlist = result

		f := Nod(ODCLFUNC, nil, nil)
		f.Func.Nname = newfuncname(name)
		f.Func.Nname.Name.Defn = f
		f.Func.Nname.Name.Param.Ntype = t // TODO: check if nname already has an ntype
		declare(f.Func.Nname, PFUNC)

		funchdr(f)
		return f

	case '(':
		// '(' oarg_type_list_ocomma ')' sym '(' oarg_type_list_ocomma ')' fnres
		rparam := p.param_list()
		name := p.sym()
		params := p.param_list()
		result := p.fnres()

		rparam = checkarglist(rparam, 0)
		params = checkarglist(params, 1)

		if rparam == nil {
			Yyerror("method has no receiver")
			return nil
		}

		if rparam.Next != nil {
			Yyerror("method has multiple receivers")
			return nil
		}

		rcvr := rparam.N
		if rcvr.Op != ODCLFIELD {
			Yyerror("bad receiver in method")
			return nil
		}

		t := Nod(OTFUNC, rcvr, nil)
		t.List = params
		t.Rlist = result

		f := Nod(ODCLFUNC, nil, nil)
		f.Func.Shortname = newfuncname(name)
		f.Func.Nname = methodname1(f.Func.Shortname, rcvr.Right)
		f.Func.Nname.Name.Defn = f
		f.Func.Nname.Name.Param.Ntype = t
		f.Func.Nname.Nointerface = nointerface
		declare(f.Func.Nname, PFUNC)

		funchdr(f)
		return f

	default:
		p.error(", expecting name or (")
		return nil
	}
}

// go.y:hidden_fndcl
func (p *parser) hidden_fndcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_fndcl")()
	}

	switch p.tok {
	default:
		// hidden_pkg_importsym '(' ohidden_funarg_list ')' ohidden_funres
		s1 := p.hidden_pkg_importsym()
		p.want('(')
		s3 := p.ohidden_funarg_list()
		p.want(')')
		s5 := p.ohidden_funres()

		s := s1
		t := functype(nil, s3, s5)

		importsym(s, ONAME)
		if s.Def != nil && s.Def.Op == ONAME {
			if Eqtype(t, s.Def.Type) {
				dclcontext = PDISCARD // since we skip funchdr below
				return nil
			}
			Yyerror("inconsistent definition for func %v during import\n\t%v\n\t%v", s, s.Def.Type, t)
		}

		ss := newfuncname(s)
		ss.Type = t
		declare(ss, PFUNC)

		funchdr(ss)
		return ss

	case '(':
		// '(' hidden_funarg_list ')' sym '(' ohidden_funarg_list ')' ohidden_funres
		p.next()
		s2 := p.hidden_funarg_list()
		p.want(')')
		s4 := p.sym()
		p.want('(')
		s6 := p.ohidden_funarg_list()
		p.want(')')
		s8 := p.ohidden_funres()

		ss := methodname1(newname(s4), s2.N.Right)
		ss.Type = functype(s2.N, s6, s8)

		checkwidth(ss.Type)
		addmethod(s4, ss.Type, false, nointerface)
		nointerface = false
		funchdr(ss)

		// inl.C's inlnode in on a dotmeth node expects to find the inlineable body as
		// (dotmeth's type).Nname.Inl, and dotmeth's type has been pulled
		// out by typecheck's lookdot as this $$.ttype.  So by providing
		// this back link here we avoid special casing there.
		ss.Type.Nname = ss
		return ss
	}
}

// go.y:fntype
func (p *parser) fntype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("fntype")()
	}

	p.want(LFUNC)
	params := p.param_list()
	result := p.fnres()

	params = checkarglist(params, 1)
	t := Nod(OTFUNC, nil, nil)
	t.List = params
	t.Rlist = result

	return t
}

// go.y:fnbody
func (p *parser) fnbody() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("fnbody")()
	}

	if p.got('{') {
		body := p.stmt_list()
		p.want('}')
		if body == nil {
			body = list1(Nod(OEMPTY, nil, nil))
		}
		return body
	}

	return nil
}

// go.y:fnres
func (p *parser) fnres() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("fnres")()
	}

	switch p.tok {
	default:
		return nil

	case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', LNAME, '@', '?':
		result := p.fnret_type()
		return list1(Nod(ODCLFIELD, nil, result))

	case '(':
		result := p.param_list()
		return checkarglist(result, 0)
	}
}

// go.y:xdcl_list
func (p *parser) xdcl_list() (l *NodeList) {
	if trace && Debug['x'] != 0 {
		defer p.trace("xdcl_list")()
	}

loop:
	for p.tok != EOF {
		switch p.tok {
		case LVAR, LCONST, LTYPE:
			l = concat(l, p.common_dcl())

		case LFUNC:
			l = list(l, p.xfndcl())

		default:
			if p.tok == '{' && l != nil && l.End.N.Op == ODCLFUNC && l.End.N.Nbody == nil {
				// opening { of function declaration on next line
				syntax_error("unexpected semicolon or newline before {")
			} else {
				syntax_error("non-declaration statement outside function body")
			}
			// skip over tokens until we find a new top-level declaration
			// TODO(gri) keep track of {} nesting as well?
			for {
				p.next()
				switch p.tok {
				case LVAR, LCONST, LTYPE, LFUNC, EOF:
					continue loop
				}
			}

		}

		if nsyntaxerrors == 0 {
			testdclstack()
		}

		noescape = false
		noinline = false
		nointerface = false
		norace = false
		nosplit = false
		nowritebarrier = false
		nowritebarrierrec = false
		systemstack = false

		// Consume ';' AFTER resetting the above flags since
		// it may read the subsequent comment line which may
		// set the flags for the next function declaration.
		if p.tok != EOF && !p.got(';') {
			p.error(" after top level declaration")
			// TODO(gri) same code above - factor!
			for {
				p.next()
				switch p.tok {
				case LVAR, LCONST, LTYPE, LFUNC, EOF:
					continue loop
				}
			}
		}
	}
	return
}

// go.y:structdcl
func (p *parser) structdcl() *NodeList {
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
			return list1(field)
		}

		// LNAME belongs to first *Sym of new_name_list
		//
		// during imports, unqualified non-exported identifiers are from builtinpkg
		if importpkg != nil && !exportname(sym.Name) {
			sym = Pkglookup(sym.Name, builtinpkg)
			if sym == nil {
				p.import_error()
			}
		}
		fallthrough

	case '@', '?':
		// new_name_list ntype oliteral
		fields := p.new_name_list(sym)
		typ := p.ntype()
		tag := p.oliteral()

		if l := fields; l == nil || l.N.Sym.Name == "?" {
			// ? symbol, during import (list1(nil) == nil)
			n := typ
			if n.Op == OIND {
				n = n.Left
			}
			n = embedded(n.Sym, importpkg)
			n.Right = typ
			n.SetVal(tag)
			return list1(n)
		}

		for l := fields; l != nil; l = l.Next {
			l.N = Nod(ODCLFIELD, l.N, typ)
			l.N.SetVal(tag)
		}
		return fields

	case '(':
		p.next()
		if p.got('*') {
			// '(' '*' embed ')' oliteral
			field := p.embed(nil)
			p.want(')')
			tag := p.oliteral()

			field.Right = Nod(OIND, field.Right, nil)
			field.SetVal(tag)
			Yyerror("cannot parenthesize embedded type")
			return list1(field)

		} else {
			// '(' embed ')' oliteral
			field := p.embed(nil)
			p.want(')')
			tag := p.oliteral()

			field.SetVal(tag)
			Yyerror("cannot parenthesize embedded type")
			return list1(field)
		}

	case '*':
		p.next()
		if p.got('(') {
			// '*' '(' embed ')' oliteral
			field := p.embed(nil)
			p.want(')')
			tag := p.oliteral()

			field.Right = Nod(OIND, field.Right, nil)
			field.SetVal(tag)
			Yyerror("cannot parenthesize embedded type")
			return list1(field)

		} else {
			// '*' embed oliteral
			field := p.embed(nil)
			tag := p.oliteral()

			field.Right = Nod(OIND, field.Right, nil)
			field.SetVal(tag)
			return list1(field)
		}

	default:
		p.error(", expecting field name or embedded type")
		return nil
	}
}

// go.y:oliteral
func (p *parser) oliteral() (v Val) {
	if p.tok == LLITERAL {
		v = p.val
		p.next()
	}
	return
}

// go.y:packname
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
		p.error(", expecting name")
		name = new(Sym)
	}

	if p.got('.') {
		// LNAME '.' sym
		s := p.sym()

		var pkg *Pkg
		if name.Def == nil || name.Def.Op != OPACK {
			Yyerror("%v is not a package", name)
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

// go.y:embed
func (p *parser) embed(sym *Sym) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("embed")()
	}

	pkgname := p.packname(sym)
	return embedded(pkgname, localpkg)
}

// go.y: interfacedcl
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
			syntax_error("name list not allowed in interface type")
		}

		if p.tok != '(' {
			// packname
			pname := p.packname(sym)
			return Nod(ODCLFIELD, nil, oldname(pname))
		}

		// newname indcl
		mname := newname(sym)
		sig := p.indcl()

		meth := Nod(ODCLFIELD, mname, sig)
		ifacedcl(meth)
		return meth

	case '(':
		p.next()
		pname := p.packname(nil)
		p.want(')')
		n := Nod(ODCLFIELD, nil, oldname(pname))
		Yyerror("cannot parenthesize embedded type")
		return n

	default:
		p.error("")
		return nil
	}
}

// go.y:indcl
func (p *parser) indcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("indcl")()
	}

	params := p.param_list()
	result := p.fnres()

	// without func keyword
	params = checkarglist(params, 1)
	t := Nod(OTFUNC, fakethis(), nil)
	t.List = params
	t.Rlist = result

	return t
}

// go.y:arg_type
func (p *parser) arg_type() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("arg_type")()
	}

	switch p.tok {
	case LNAME, '@', '?':
		s1 := p.sym()
		switch p.tok {
		case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', LNAME, '@', '?', '(':
			// sym name_or_type
			s2 := p.ntype()
			ss := Nod(ONONAME, nil, nil)
			ss.Sym = s1
			return Nod(OKEY, ss, s2)

		case LDDD:
			// sym dotdotdot
			s2 := p.dotdotdot()
			ss := Nod(ONONAME, nil, nil)
			ss.Sym = s1
			return Nod(OKEY, ss, s2)

		default:
			// name_or_type
			s1 := mkname(s1)
			// from dotname
			if p.got('.') {
				s3 := p.sym()

				if s1.Op == OPACK {
					var s *Sym
					s = restrictlookup(s3.Name, s1.Name.Pkg)
					s1.Used = true
					return oldname(s)
				}
				return Nod(OXDOT, s1, newname(s3))
			}
			return s1
		}

	case LDDD:
		// dotdotdot
		return p.dotdotdot()

	case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', '(':
		// name_or_type
		return p.ntype()

	default:
		p.error(", expecting )")
		return nil
	}
}

// go.y:oarg_type_list_ocomma + surrounding ()'s
func (p *parser) param_list() (l *NodeList) {
	if trace && Debug['x'] != 0 {
		defer p.trace("param_list")()
	}

	p.want('(')
	for p.tok != EOF && p.tok != ')' {
		l = list(l, p.arg_type())
		p.ocomma("parameter list")
	}
	p.want(')')
	return
}

var missing_stmt = Nod(OXXX, nil, nil)

// go.y:stmt
// maty return missing_stmt
func (p *parser) stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("stmt")()
	}

	switch p.tok {
	case '{':
		return p.compound_stmt(false)

	case LVAR, LCONST, LTYPE:
		return liststmt(p.common_dcl())

	case LNAME, '@', '?', LLITERAL, LFUNC, '(', // operands
		'[', LSTRUCT, LMAP, LCHAN, LINTERFACE, // composite types
		'+', '-', '*', '&', '^', '~', LCOMM, '!': // unary operators
		// simple_stmt
		fallthrough

	case LFOR, LSWITCH, LSELECT, LIF, LFALL, LBREAK, LCONTINUE, LGO, LDEFER, LGOTO, LRETURN:
		return p.non_dcl_stmt()

	case ';':
		return nil

	default:
		return missing_stmt
	}
}

// TODO(gri) inline non_dcl_stmt into stmt
// go.y:non_dcl_stmt
func (p *parser) non_dcl_stmt() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("non_dcl_stmt")()
	}

	switch p.tok {
	case LNAME, '@', '?', LLITERAL, LFUNC, '(', // operands
		'[', LSTRUCT, LMAP, LCHAN, LINTERFACE, // composite types
		'+', '-', '*', '&', '^', '~', LCOMM, '!': // unary operators
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
		ss := Nod(OXFALL, nil, nil)
		ss.Xoffset = int64(block)
		return ss

	case LBREAK:
		p.next()
		s2 := p.onew_name()

		return Nod(OBREAK, s2, nil)

	case LCONTINUE:
		p.next()
		s2 := p.onew_name()

		return Nod(OCONTINUE, s2, nil)

	case LGO:
		p.next()
		s2 := p.pseudocall()

		return Nod(OPROC, s2, nil)

	case LDEFER:
		p.next()
		s2 := p.pseudocall()

		return Nod(ODEFER, s2, nil)

	case LGOTO:
		p.next()
		s2 := p.new_name(p.sym())

		ss := Nod(OGOTO, s2, nil)
		ss.Sym = dclstack // context, for goto restrictions
		return ss

	case LRETURN:
		p.next()
		var s2 *NodeList
		if p.tok != ';' && p.tok != '}' {
			s2 = p.expr_list()
		}

		ss := Nod(ORETURN, nil, nil)
		ss.List = s2
		if ss.List == nil && Curfn != nil {
			var l *NodeList

			for l = Curfn.Func.Dcl; l != nil; l = l.Next {
				if l.N.Class == PPARAM {
					continue
				}
				if l.N.Class != PPARAMOUT {
					break
				}
				if l.N.Sym.Def != l.N {
					Yyerror("%s is shadowed during return", l.N.Sym.Name)
				}
			}
		}

		return ss

	default:
		panic("unreachable")
	}
}

// go.y:stmt_list
func (p *parser) stmt_list() (l *NodeList) {
	if trace && Debug['x'] != 0 {
		defer p.trace("stmt_list")()
	}

	for p.tok != EOF && p.tok != '}' && p.tok != LCASE && p.tok != LDEFAULT {
		s := p.stmt()
		if s == missing_stmt {
			break
		}
		l = list(l, s)
		// customized version of osemi:
		// ';' is optional before a closing ')' or '}'
		if p.tok == ')' || p.tok == '}' {
			continue
		}
		if !p.got(';') {
			p.error(" at end of statement")
		}
	}
	return
}

// go.y:new_name_list
// if first != nil we have the first symbol already
func (p *parser) new_name_list(first *Sym) *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("new_name_list")()
	}

	if first == nil {
		first = p.sym() // may still be nil
	}
	l := list1(p.new_name(first))
	for p.got(',') {
		l = list(l, p.new_name(p.sym()))
	}
	return l
}

// go.y:dcl_name_list
func (p *parser) dcl_name_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("dcl_name_list")()
	}

	l := list1(dclname(p.sym()))
	for p.got(',') {
		l = list(l, dclname(p.sym()))
	}
	return l
}

// go.y:expr_list
func (p *parser) expr_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("expr_list")()
	}

	l := list1(p.expr())
	for p.got(',') {
		l = list(l, p.expr())
	}
	return l
}

// go.y:expr_or_type_list
func (p *parser) arg_list() (l *NodeList, ddd bool) {
	if trace && Debug['x'] != 0 {
		defer p.trace("arg_list")()
	}

	// TODO(gri) make this more tolerant in the presence of LDDD
	// that is not at the end.

	for p.tok != EOF && p.tok != ')' && !ddd {
		l = list(l, p.expr()) // expr_or_type
		ddd = p.got(LDDD)
		p.ocomma("argument list")
	}

	return
}

// go.y:osemi
func (p *parser) osemi() {
	// ';' is optional before a closing ')' or '}'
	if p.tok == ')' || p.tok == '}' {
		return
	}
	p.want(';')
}

// go.y:ocomma
func (p *parser) ocomma(context string) {
	switch p.tok {
	case ')', '}':
		// ',' is optional before a closing ')' or '}'
		return
	case ';':
		syntax_error("need trailing comma before newline in " + context)
		p.next()
		return
	}
	p.want(',')
}

// ----------------------------------------------------------------------------
// Importing packages

func (p *parser) import_error() {
	p.error(" in export data of imported package")
}

// The methods below reflect a 1:1 translation of the corresponding go.y yacc
// productions They could be simplified significantly and also use better
// variable names. However, we will be able to delete them once we enable the
// new export format by default, so it's not worth the effort.

// go.y:hidden_importsym:
func (p *parser) hidden_importsym() *Sym {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_importsym")()
	}

	p.want('@')
	var s2 Val
	if p.tok == LLITERAL {
		s2 = p.val
		p.next()
	} else {
		p.import_error()
	}
	p.want('.')

	switch p.tok {
	case LNAME:
		s4 := p.sym_
		p.next()

		var p *Pkg

		if s2.U.(string) == "" {
			p = importpkg
		} else {
			if isbadimport(s2.U.(string)) {
				errorexit()
			}
			p = mkpkg(s2.U.(string))
		}
		return Pkglookup(s4.Name, p)

	case '?':
		p.next()

		var p *Pkg

		if s2.U.(string) == "" {
			p = importpkg
		} else {
			if isbadimport(s2.U.(string)) {
				errorexit()
			}
			p = mkpkg(s2.U.(string))
		}
		return Pkglookup("?", p)

	default:
		p.import_error()
		return nil
	}
}

// go.y:ohidden_funarg_list
func (p *parser) ohidden_funarg_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("ohidden_funarg_list")()
	}

	var ss *NodeList
	if p.tok != ')' {
		ss = p.hidden_funarg_list()
	}
	return ss
}

// go.y:ohidden_structdcl_list
func (p *parser) ohidden_structdcl_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("ohidden_structdcl_list")()
	}

	var ss *NodeList
	if p.tok != '}' {
		ss = p.hidden_structdcl_list()
	}
	return ss
}

// go.y:ohidden_interfacedcl_list
func (p *parser) ohidden_interfacedcl_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("ohidden_interfacedcl_list")()
	}

	var ss *NodeList
	if p.tok != '}' {
		ss = p.hidden_interfacedcl_list()
	}
	return ss
}

// import syntax from package header
//
// go.y:hidden_import
func (p *parser) hidden_import() {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_import")()
	}

	switch p.tok {
	case LIMPORT:
		// LIMPORT LNAME LLITERAL ';'
		p.next()
		var s2 *Sym
		if p.tok == LNAME {
			s2 = p.sym_
			p.next()
		} else {
			p.import_error()
		}
		var s3 Val
		if p.tok == LLITERAL {
			s3 = p.val
			p.next()
		} else {
			p.import_error()
		}
		p.want(';')

		importimport(s2, s3.U.(string))

	case LVAR:
		// LVAR hidden_pkg_importsym hidden_type ';'
		p.next()
		s2 := p.hidden_pkg_importsym()
		s3 := p.hidden_type()
		p.want(';')

		importvar(s2, s3)

	case LCONST:
		// LCONST hidden_pkg_importsym '=' hidden_constant ';'
		// LCONST hidden_pkg_importsym hidden_type '=' hidden_constant ';'
		p.next()
		s2 := p.hidden_pkg_importsym()
		var s3 *Type = Types[TIDEAL]
		if p.tok != '=' {
			s3 = p.hidden_type()
		}
		p.want('=')
		s4 := p.hidden_constant()
		p.want(';')

		importconst(s2, s3, s4)

	case LTYPE:
		// LTYPE hidden_pkgtype hidden_type ';'
		p.next()
		s2 := p.hidden_pkgtype()
		s3 := p.hidden_type()
		p.want(';')

		importtype(s2, s3)

	case LFUNC:
		// LFUNC hidden_fndcl fnbody ';'
		p.next()
		s2 := p.hidden_fndcl()
		s3 := p.fnbody()
		p.want(';')

		if s2 == nil {
			dclcontext = PEXTERN // since we skip the funcbody below
			return
		}

		s2.Func.Inl = s3

		funcbody(s2)
		importlist = append(importlist, s2)

		if Debug['E'] > 0 {
			fmt.Printf("import [%q] func %v \n", importpkg.Path, s2)
			if Debug['m'] > 2 && s2.Func.Inl != nil {
				fmt.Printf("inl body:%v\n", s2.Func.Inl)
			}
		}

	default:
		p.import_error()
	}
}

// go.y:hidden_pkg_importsym
func (p *parser) hidden_pkg_importsym() *Sym {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_pkg_importsym")()
	}

	s1 := p.hidden_importsym()

	ss := s1
	structpkg = ss.Pkg

	return ss
}

// go.y:hidden_pkgtype
func (p *parser) hidden_pkgtype() *Type {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_pkgtype")()
	}

	s1 := p.hidden_pkg_importsym()

	ss := pkgtype(s1)
	importsym(s1, OTYPE)

	return ss
}

// ----------------------------------------------------------------------------
// Importing types

// go.y:hidden_type
func (p *parser) hidden_type() *Type {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_type")()
	}

	switch p.tok {
	default:
		return p.hidden_type_misc()
	case LCOMM:
		return p.hidden_type_recv_chan()
	case LFUNC:
		return p.hidden_type_func()
	}
}

// go.y:hidden_type_non_recv_chan
func (p *parser) hidden_type_non_recv_chan() *Type {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_type_non_recv_chan")()
	}

	switch p.tok {
	default:
		return p.hidden_type_misc()
	case LFUNC:
		return p.hidden_type_func()
	}
}

// go.y:hidden_type_misc
func (p *parser) hidden_type_misc() *Type {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_type_misc")()
	}

	switch p.tok {
	case '@':
		// hidden_importsym
		s1 := p.hidden_importsym()
		return pkgtype(s1)

	case LNAME:
		// LNAME
		s1 := p.sym_
		p.next()

		// predefined name like uint8
		s1 = Pkglookup(s1.Name, builtinpkg)
		if s1.Def == nil || s1.Def.Op != OTYPE {
			Yyerror("%s is not a type", s1.Name)
			return nil
		} else {
			return s1.Def.Type
		}

	case '[':
		// '[' ']' hidden_type
		// '[' LLITERAL ']' hidden_type
		p.next()
		var s2 *Node
		if p.tok == LLITERAL {
			s2 = nodlit(p.val)
			p.next()
		}
		p.want(']')
		s4 := p.hidden_type()

		return aindex(s2, s4)

	case LMAP:
		// LMAP '[' hidden_type ']' hidden_type
		p.next()
		p.want('[')
		s3 := p.hidden_type()
		p.want(']')
		s5 := p.hidden_type()

		return maptype(s3, s5)

	case LSTRUCT:
		// LSTRUCT '{' ohidden_structdcl_list '}'
		p.next()
		p.want('{')
		s3 := p.ohidden_structdcl_list()
		p.want('}')

		return tostruct(s3)

	case LINTERFACE:
		// LINTERFACE '{' ohidden_interfacedcl_list '}'
		p.next()
		p.want('{')
		s3 := p.ohidden_interfacedcl_list()
		p.want('}')

		return tointerface(s3)

	case '*':
		// '*' hidden_type
		p.next()
		s2 := p.hidden_type()
		return Ptrto(s2)

	case LCHAN:
		p.next()
		switch p.tok {
		default:
			// LCHAN hidden_type_non_recv_chan
			s2 := p.hidden_type_non_recv_chan()
			ss := typ(TCHAN)
			ss.Type = s2
			ss.Chan = Cboth
			return ss

		case '(':
			// LCHAN '(' hidden_type_recv_chan ')'
			p.next()
			s3 := p.hidden_type_recv_chan()
			p.want(')')
			ss := typ(TCHAN)
			ss.Type = s3
			ss.Chan = Cboth
			return ss

		case LCOMM:
			// LCHAN hidden_type
			p.next()
			s3 := p.hidden_type()
			ss := typ(TCHAN)
			ss.Type = s3
			ss.Chan = Csend
			return ss
		}

	default:
		p.import_error()
		return nil
	}
}

// go.y:hidden_type_recv_chan
func (p *parser) hidden_type_recv_chan() *Type {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_type_recv_chan")()
	}

	p.want(LCOMM)
	p.want(LCHAN)
	s3 := p.hidden_type()

	ss := typ(TCHAN)
	ss.Type = s3
	ss.Chan = Crecv
	return ss
}

// go.y:hidden_type_func
func (p *parser) hidden_type_func() *Type {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_type_func")()
	}

	p.want(LFUNC)
	p.want('(')
	s3 := p.ohidden_funarg_list()
	p.want(')')
	s5 := p.ohidden_funres()

	return functype(nil, s3, s5)
}

// go.y:hidden_funarg
func (p *parser) hidden_funarg() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_funarg")()
	}

	s1 := p.sym()
	switch p.tok {
	default:
		s2 := p.hidden_type()
		s3 := p.oliteral()

		ss := Nod(ODCLFIELD, nil, typenod(s2))
		if s1 != nil {
			ss.Left = newname(s1)
		}
		ss.SetVal(s3)
		return ss

	case LDDD:
		p.next()
		s3 := p.hidden_type()
		s4 := p.oliteral()

		var t *Type

		t = typ(TARRAY)
		t.Bound = -1
		t.Type = s3

		ss := Nod(ODCLFIELD, nil, typenod(t))
		if s1 != nil {
			ss.Left = newname(s1)
		}
		ss.Isddd = true
		ss.SetVal(s4)

		return ss
	}
}

// go.y:hidden_structdcl
func (p *parser) hidden_structdcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_structdcl")()
	}

	s1 := p.sym()
	s2 := p.hidden_type()
	s3 := p.oliteral()

	var s *Sym
	var pkg *Pkg

	var ss *Node
	if s1 != nil && s1.Name != "?" {
		ss = Nod(ODCLFIELD, newname(s1), typenod(s2))
		ss.SetVal(s3)
	} else {
		s = s2.Sym
		if s == nil && Isptr[s2.Etype] {
			s = s2.Type.Sym
		}
		pkg = importpkg
		if s1 != nil {
			pkg = s1.Pkg
		}
		ss = embedded(s, pkg)
		ss.Right = typenod(s2)
		ss.SetVal(s3)
	}

	return ss
}

// go.y:hidden_interfacedcl
func (p *parser) hidden_interfacedcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_interfacedcl")()
	}

	// TODO(gri) possible conflict here: both cases may start with '@' per grammar
	switch p.tok {
	case LNAME, '@', '?':
		s1 := p.sym()
		p.want('(')
		s3 := p.ohidden_funarg_list()
		p.want(')')
		s5 := p.ohidden_funres()

		return Nod(ODCLFIELD, newname(s1), typenod(functype(fakethis(), s3, s5)))

	default:
		s1 := p.hidden_type()

		return Nod(ODCLFIELD, nil, typenod(s1))
	}
}

// go.y:ohidden_funres
func (p *parser) ohidden_funres() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("ohidden_funres")()
	}

	switch p.tok {
	default:
		return nil

	case '(', '@', LNAME, '[', LMAP, LSTRUCT, LINTERFACE, '*', LCHAN, LCOMM, LFUNC:
		return p.hidden_funres()
	}
}

// go.y:hidden_funres
func (p *parser) hidden_funres() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_funres")()
	}

	switch p.tok {
	case '(':
		p.next()
		s2 := p.ohidden_funarg_list()
		p.want(')')
		return s2

	default:
		s1 := p.hidden_type()
		return list1(Nod(ODCLFIELD, nil, typenod(s1)))
	}
}

// ----------------------------------------------------------------------------
// Importing constants

// go.y:hidden_literal
func (p *parser) hidden_literal() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_literal")()
	}

	switch p.tok {
	case LLITERAL:
		ss := nodlit(p.val)
		p.next()
		return ss

	case '-':
		p.next()
		if p.tok == LLITERAL {
			ss := nodlit(p.val)
			p.next()
			switch ss.Val().Ctype() {
			case CTINT, CTRUNE:
				mpnegfix(ss.Val().U.(*Mpint))
				break
			case CTFLT:
				mpnegflt(ss.Val().U.(*Mpflt))
				break
			case CTCPLX:
				mpnegflt(&ss.Val().U.(*Mpcplx).Real)
				mpnegflt(&ss.Val().U.(*Mpcplx).Imag)
				break
			default:
				Yyerror("bad negated constant")
			}
			return ss
		} else {
			p.import_error()
			return nil
		}

	case LNAME, '@', '?':
		s1 := p.sym()
		ss := oldname(Pkglookup(s1.Name, builtinpkg))
		if ss.Op != OLITERAL {
			Yyerror("bad constant %v", ss.Sym)
		}
		return ss

	default:
		p.import_error()
		return nil
	}
}

// go.y:hidden_constant
func (p *parser) hidden_constant() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_constant")()
	}

	switch p.tok {
	default:
		return p.hidden_literal()
	case '(':
		p.next()
		s2 := p.hidden_literal()
		p.want('+')
		s4 := p.hidden_literal()
		p.want(')')

		if s2.Val().Ctype() == CTRUNE && s4.Val().Ctype() == CTINT {
			ss := s2
			mpaddfixfix(s2.Val().U.(*Mpint), s4.Val().U.(*Mpint), 0)
			return ss
		}
		s4.Val().U.(*Mpcplx).Real = s4.Val().U.(*Mpcplx).Imag
		Mpmovecflt(&s4.Val().U.(*Mpcplx).Imag, 0.0)
		return nodcplxlit(s2.Val(), s4.Val())
	}
}

// go.y:hidden_import_list
func (p *parser) hidden_import_list() {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_import_list")()
	}

	for p.tok != '$' {
		p.hidden_import()
	}
}

// go.y:hidden_funarg_list
func (p *parser) hidden_funarg_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_funarg_list")()
	}

	s1 := p.hidden_funarg()
	ss := list1(s1)
	for p.got(',') {
		s3 := p.hidden_funarg()
		ss = list(ss, s3)
	}
	return ss
}

// go.y:hidden_structdcl_list
func (p *parser) hidden_structdcl_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_structdcl_list")()
	}

	s1 := p.hidden_structdcl()
	ss := list1(s1)
	for p.got(';') {
		s3 := p.hidden_structdcl()
		ss = list(ss, s3)
	}
	return ss
}

// go.y:hidden_interfacedcl_list
func (p *parser) hidden_interfacedcl_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_interfacedcl_list")()
	}

	s1 := p.hidden_interfacedcl()
	ss := list1(s1)
	for p.got(';') {
		s3 := p.hidden_interfacedcl()
		ss = list(ss, s3)
	}
	return ss
}
