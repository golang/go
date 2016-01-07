// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// The recursive-descent parser is built around a slighty modified grammar
// of Go to accomodate for the constraints imposed by strict one token look-
// ahead, and for better error handling. Subsequent checks of the constructed
// syntax tree restrict the language accepted by the compiler to proper Go.
//
// Semicolons are inserted by the lexer. The parser uses one-token look-ahead
// to handle optional commas and semicolons before a closing ) or } .

import (
	"fmt"
	"strconv"
	"strings"
)

const trace = false // if set, parse tracing can be enabled with -x

// TODO(gri) Once we handle imports w/o redirecting the underlying
// source of the lexer we can get rid of these. They are here for
// compatibility with the existing yacc-based parser setup (issue 13242).
var thenewparser parser // the parser in use
var savedstate []parser // saved parser state, used during import

func push_parser() {
	// Indentation (for tracing) must be preserved across parsers
	// since we are changing the lexer source (and parser state)
	// under foot, in the middle of productions. This won't be
	// needed anymore once we fix issue 13242, but neither will
	// be the push/pop_parser functionality.
	// (Instead we could just use a global variable indent, but
	// but eventually indent should be parser-specific anyway.)
	indent := thenewparser.indent
	savedstate = append(savedstate, thenewparser)
	thenewparser = parser{indent: indent} // preserve indentation
	thenewparser.next()
}

func pop_parser() {
	indent := thenewparser.indent
	n := len(savedstate) - 1
	thenewparser = savedstate[n]
	thenewparser.indent = indent // preserve indentation
	savedstate = savedstate[:n]
}

// parse_file sets up a new parser and parses a single Go source file.
func parse_file() {
	thenewparser = parser{}
	thenewparser.loadsys()
	thenewparser.next()
	thenewparser.file()
}

// loadsys loads the definitions for the low-level runtime functions,
// so that the compiler can generate calls to them,
// but does not make the name "runtime" visible as a package.
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
	fnest  int       // function nesting level (for error handling)
	xnest  int       // expression nesting level (for complit ambiguity resolution)
	yy     yySymType // for temporary use by next
	indent []byte    // tracing support
}

func (p *parser) next() {
	p.tok = yylex(&p.yy)
	p.op = p.yy.op
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
		Yyerror("syntax error: %s", msg)
		return
	}

	// determine token string
	var tok string
	switch p.tok {
	case LLITERAL:
		tok = litbuf
	case LNAME:
		if p.sym_ != nil && p.sym_.Name != "" {
			tok = p.sym_.Name
		} else {
			tok = "name"
		}
	case LASOP:
		tok = goopnames[p.op] + "="
	default:
		tok = tokstring(p.tok)
	}

	Yyerror("syntax error: unexpected %s", tok+msg)
}

// Like syntax_error, but reports error at given line rather than current lexer line.
func (p *parser) syntax_error_at(lineno int32, msg string) {
	defer func(lineno int32) {
		lexlineno = lineno
	}(lexlineno)
	lexlineno = lineno
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
	LLITERAL:   "LLITERAL",
	LASOP:      "op=",
	LCOLAS:     ":=",
	LBREAK:     "break",
	LCASE:      "case",
	LCHAN:      "chan",
	LCONST:     "const",
	LCONTINUE:  "continue",
	LDDD:       "...",
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
	LNAME:      "LNAME",
	LPACKAGE:   "package",
	LRANGE:     "range",
	LRETURN:    "return",
	LSELECT:    "select",
	LSTRUCT:    "struct",
	LSWITCH:    "switch",
	LTYPE:      "type",
	LVAR:       "var",
	LANDAND:    "&&",
	LANDNOT:    "&^",
	LCOMM:      "<-",
	LDEC:       "--",
	LEQ:        "==",
	LGE:        ">=",
	LGT:        ">",
	LIGNORE:    "LIGNORE", // we should never see this one
	LINC:       "++",
	LLE:        "<=",
	LLSH:       "<<",
	LLT:        "<",
	LNE:        "!=",
	LOROR:      "||",
	LRSH:       ">>",
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

	xtop = concat(xtop, p.xdcl_list())

	p.want(EOF)
}

// PackageClause = "package" PackageName .
// PackageName   = identifier .
func (p *parser) package_() {
	if trace && Debug['x'] != 0 {
		defer p.trace("package_")()
	}

	if p.got(LPACKAGE) {
		mkpackage(p.sym().Name)
	} else {
		prevlineno = lineno // see issue #13267
		p.syntax_error("package statement must be first")
		errorexit()
	}
}

// ImportDecl = "import" ( ImportSpec | "(" { ImportSpec ";" } ")" ) .
func (p *parser) import_() {
	if trace && Debug['x'] != 0 {
		defer p.trace("import_")()
	}

	p.want(LIMPORT)
	if p.got('(') {
		for p.tok != EOF && p.tok != ')' {
			p.import_stmt()
			if !p.osemi(')') {
				break
			}
		}
		p.want(')')
	} else {
		p.import_stmt()
	}
}

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

// ImportSpec = [ "." | PackageName ] ImportPath .
// ImportPath = string_lit .
//
// import_here switches the underlying lexed source to the export data
// of the imported package.
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
		p.syntax_error("missing import path; require quoted string")
		p.advance(';', ')')
	}

	line := parserline()
	importfile(&path, line)
	return line
}

// import_package parses the header of an imported package as exported
// in textual format from another package.
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

// import_there parses the imported package definitions and then switches
// the underlying lexed source back to the importing package.
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

// Declaration = ConstDecl | TypeDecl | VarDecl .
// ConstDecl   = "const" ( ConstSpec | "(" { ConstSpec ";" } ")" ) .
// TypeDecl    = "type" ( TypeSpec | "(" { TypeSpec ";" } ")" ) .
// VarDecl     = "var" ( VarSpec | "(" { VarSpec ";" } ")" ) .
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
			if !p.osemi(')') {
				break
			}
		}
		p.want(')')
	} else {
		l = dcl()
	}

	iota_ = -100000
	lastconst = nil

	return l
}

// VarSpec = IdentifierList ( Type [ "=" ExpressionList ] | "=" ExpressionList ) .
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

// ConstSpec = IdentifierList [ [ Type ] "=" ExpressionList ] .
func (p *parser) constdcl() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("constdcl")()
	}

	names := p.dcl_name_list()
	var typ *Node
	var exprs *NodeList
	if p.tok != EOF && p.tok != ';' && p.tok != ')' {
		typ = p.try_ntype()
		if p.got('=') {
			exprs = p.expr_list()
		}
	}

	return constiter(names, typ, exprs)
}

// TypeSpec = identifier Type .
func (p *parser) typedcl() *NodeList {
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

	return list1(typedcl1(name, typ, true))
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
				// (calling p.name()) and given an ONAME, ONONAME, OTYPE, OPACK, or OLITERAL node.
				switch lhs.Op {
				case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
					lhs = newname(lhs.Sym)
				default:
					p.syntax_error("expecting semicolon or newline or }")
					// we already progressed, no need to advance
				}
				lhs := Nod(OLABEL, lhs, nil)
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
		lno := lineno
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
			ts := Nod(OTYPESW, nil, rhs.N.Right)
			if rhs.Next != nil {
				Yyerror("expr.(type) must be alone in list")
			}
			if lhs.Next != nil {
				Yyerror("argument count mismatch: %d = %d", count(lhs), 1)
			} else if (lhs.N.Op != ONAME && lhs.N.Op != OTYPE && lhs.N.Op != ONONAME && (lhs.N.Op != OLITERAL || lhs.N.Name == nil)) || isblank(lhs.N) {
				Yyerror("invalid variable name %s in type switch", lhs.N)
			} else {
				ts.Left = dclname(lhs.N.Sym)
			} // it's a colas, so must not re-use an oldname
			return ts
		}
		return colas(lhs, rhs, int32(lno))

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
			p.syntax_error_at(prevlineno, "missing statement after label")
			// we are already at the end of the labeled statement - no need to advance
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
			markdcl() // matching popdcl in caseblock
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
			lno := lineno
			p.next()
			rhs := p.expr()

			// will be converted to OCASE
			// right will point to next case
			// done in casebody()
			markdcl() // matching popdcl in caseblock
			stmt := Nod(OXCASE, nil, nil)
			stmt.List = list1(colas(cases, list1(rhs), int32(lno)))

			p.want(':') // consume ':' after declaring select cases for correct lineno
			return stmt

		default:
			markdcl()                     // for matching popdcl in caseblock
			stmt := Nod(OXCASE, nil, nil) // don't return nil
			p.syntax_error("expecting := or = or : or comma")
			p.advance(LCASE, LDEFAULT, '}')
			return stmt
		}

	case LDEFAULT:
		// LDEFAULT ':'
		p.next()

		markdcl() // matching popdcl in caseblock
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
		markdcl()                     // matching popdcl in caseblock
		stmt := Nod(OXCASE, nil, nil) // don't return nil
		p.syntax_error("expecting case or default or }")
		p.advance(LCASE, LDEFAULT, '}')
		return stmt
	}
}

// Block         = "{" StatementList "}" .
// StatementList = { Statement ";" } .
func (p *parser) compound_stmt(else_clause bool) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("compound_stmt")()
	}

	markdcl()
	if p.got('{') {
		// ok
	} else if else_clause {
		p.syntax_error("else must be followed by if or statement block")
		p.advance(LNAME, '}')
	} else {
		panic("unreachable")
	}

	l := p.stmt_list()
	p.want('}')

	var stmt *Node
	if l == nil {
		stmt = Nod(OEMPTY, nil, nil)
	} else {
		stmt = liststmt(l)
	}
	popdcl()

	return stmt
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
	stmt.Nbody = p.stmt_list()

	popdcl()

	return stmt
}

// caseblock_list parses a superset of switch and select clause lists.
func (p *parser) caseblock_list(tswitch *Node) (l *NodeList) {
	if trace && Debug['x'] != 0 {
		defer p.trace("caseblock_list")()
	}

	if !p.got('{') {
		p.syntax_error("missing { after switch clause")
		p.advance(LCASE, LDEFAULT, '}')
	}

	for p.tok != EOF && p.tok != '}' {
		l = list(l, p.caseblock(tswitch))
	}
	p.want('}')
	return
}

// loop_body parses if and for statement bodies.
func (p *parser) loop_body(context string) *NodeList {
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

func (p *parser) for_body() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("for_body")()
	}

	stmt := p.for_header()
	body := p.loop_body("for clause")

	stmt.Nbody = concat(stmt.Nbody, body)
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
			Yyerror("var declaration not allowed in for initializer")
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
	h := Nod(OIF, nil, nil)
	h.Ninit = list1(init)
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
		Yyerror("missing condition in if statement")
	}

	stmt.Nbody = p.loop_body("if clause")

	l := p.elseif_list_else() // does markdcl

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

func (p *parser) elseif() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("elseif")()
	}

	// LELSE LIF already consumed
	markdcl() // matching popdcl in if_stmt

	stmt := p.if_header()
	if stmt.Left == nil {
		Yyerror("missing condition in if statement")
	}

	stmt.Nbody = p.loop_body("if clause")

	return list1(stmt)
}

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

func (p *parser) else_() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("else")()
	}

	l := &NodeList{N: p.compound_stmt(true)}
	l.End = l
	return l

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

	hdr.List = p.caseblock_list(tswitch)
	popdcl()

	return hdr
}

// SelectStmt = "select" "{" { CommClause } "}" .
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
// (issue 13244)
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

// Expression = UnaryExpr | Expression binary_op Expression .
func (p *parser) bexpr(prec int) *Node {
	// don't trace bexpr - only leads to overly nested trace output

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

func (p *parser) expr() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("expr")()
	}

	return p.bexpr(1)
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
		// TODO(gri) do this in the lexer instead (issue 13244)
		p.next()
		x := p.uexpr()
		Yyerror("the bitwise complement operator is ^")
		return Nod(OCOM, x, nil)

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
			dir := EType(Csend)
			t := x
			for ; t.Op == OTCHAN && dir == Csend; t = t.Left {
				dir = t.Etype
				if dir == Crecv {
					// t is type <-chan E but <-<-chan E is not permitted
					// (report same error as for "type _ <-<-chan E")
					p.syntax_error("unexpected <-, expecting chan")
					// already progressed, no need to advance
				}
				t.Etype = Crecv
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
		return Nod(ORECV, x, nil)

	default:
		return p.pexpr(false)
	}

	// simple uexpr
	p.next()
	return Nod(op, p.uexpr(), nil)
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
		Yyerror("expression in go/defer must not be parenthesized")
		// already progressed, no need to advance
	default:
		Yyerror("expression in go/defer must be function call")
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

	case LNAME, '@', '?':
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
			x = Nod(OPAREN, x, nil)
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
			case LNAME, '@', '?':
				// pexpr '.' sym
				x = p.new_dotname(x)

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
			args, ddd := p.arg_list()

			// call or conversion
			x = Nod(OCALL, x, nil)
			x.List = args
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
		return Nod(OKEY, x, wrapname(p.bare_complitexpr()))
	}

	// value
	return wrapname(x)
}

func wrapname(x *Node) *Node {
	// These nodes do not carry line numbers.
	// Introduce a wrapper node to give the correct line.
	switch x.Op {
	case ONAME, ONONAME, OTYPE, OPACK, OLITERAL:
		x = Nod(OPAREN, x, nil)
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
	n := Nod(OCOMPLIT, nil, nil)

	p.want('{')
	p.xnest++

	var l *NodeList
	for p.tok != EOF && p.tok != '}' {
		l = list(l, p.keyval())
		if !p.ocomma('}') {
			break
		}
	}

	p.xnest--
	p.want('}')

	n.List = l
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

func (p *parser) dcl_name(sym *Sym) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("dcl_name")()
	}

	if sym == nil {
		yyerrorl(int(prevlineno), "invalid declaration")
		return nil
	}
	return dclname(sym)
}

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
		p.syntax_error("expecting name")
		p.advance()
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
		return Nod(ODDD, typ, nil)
	}

	Yyerror("final argument in variadic function missing type")
	return Nod(ODDD, typenod(typ(TINTER)), nil)
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
		t := Nod(OTCHAN, p.chan_elem(), nil)
		t.Etype = Crecv
		return t

	case LFUNC:
		// fntype
		p.next()
		params := p.param_list()
		result := p.fnres()
		params = checkarglist(params, 1)
		t := Nod(OTFUNC, nil, nil)
		t.List = params
		t.Rlist = result
		return t

	case '[':
		// '[' oexpr ']' ntype
		// '[' LDDD ']' ntype
		p.next()
		p.xnest++
		var len *Node
		if p.tok != ']' {
			if p.got(LDDD) {
				len = Nod(ODDD, nil, nil)
			} else {
				len = p.expr()
			}
		}
		p.xnest--
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
		return p.structtype()

	case LINTERFACE:
		return p.interfacetype()

	case '*':
		// ptrtype
		p.next()
		return Nod(OIND, p.ntype(), nil)

	case LNAME, '@', '?':
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

func (p *parser) new_dotname(pkg *Node) *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("new_dotname")()
	}

	sel := p.sym()
	if pkg.Op == OPACK {
		s := restrictlookup(sel.Name, pkg.Name.Pkg)
		pkg.Used = true
		return oldname(s)
	}
	return Nod(OXDOT, pkg, newname(sel))

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
	var l *NodeList
	for p.tok != EOF && p.tok != '}' {
		l = concat(l, p.structdcl())
		if !p.osemi('}') {
			break
		}
	}
	p.want('}')

	t := Nod(OTSTRUCT, nil, nil)
	t.List = l
	return t
}

// InterfaceType = "interface" "{" { MethodSpec ";" } "}" .
func (p *parser) interfacetype() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("interfacetype")()
	}

	p.want(LINTERFACE)
	p.want('{')
	var l *NodeList
	for p.tok != EOF && p.tok != '}' {
		l = list(l, p.interfacedcl())
		if !p.osemi('}') {
			break
		}
	}
	p.want('}')

	t := Nod(OTINTER, nil, nil)
	t.List = l
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
		p.syntax_error("expecting name or (")
		p.advance('{', ';')
		return nil
	}
}

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

// FunctionBody = Block .
func (p *parser) fnbody() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("fnbody")()
	}

	if p.got('{') {
		p.fnest++
		body := p.stmt_list()
		p.fnest--
		p.want('}')
		if body == nil {
			body = list1(Nod(OEMPTY, nil, nil))
		}
		return body
	}

	return nil
}

// Result = Parameters | Type .
func (p *parser) fnres() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("fnres")()
	}

	if p.tok == '(' {
		result := p.param_list()
		return checkarglist(result, 0)
	}

	if result := p.try_ntype(); result != nil {
		return list1(Nod(ODCLFIELD, nil, result))
	}

	return nil
}

// Declaration  = ConstDecl | TypeDecl | VarDecl .
// TopLevelDecl = Declaration | FunctionDecl | MethodDecl .
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
				p.syntax_error("unexpected semicolon or newline before {")
			} else {
				p.syntax_error("non-declaration statement outside function body")
			}
			p.advance(LVAR, LCONST, LTYPE, LFUNC)
			goto loop
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
			p.syntax_error("after top level declaration")
			p.advance(LVAR, LCONST, LTYPE, LFUNC)
			goto loop
		}
	}
	return
}

// FieldDecl      = (IdentifierList Type | AnonymousField) [ Tag ] .
// AnonymousField = [ "*" ] TypeName .
// Tag            = string_lit .
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
		p.syntax_error("")
		p.advance(';', '}')
		return nil
	}
}

// MethodSpec = MethodName Signature .
// MethodName = identifier .
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

// ParameterDecl = [ IdentifierList ] [ "..." ] Type .
func (p *parser) arg_type() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("arg_type")()
	}

	switch p.tok {
	case LNAME, '@', '?':
		name := p.sym()
		switch p.tok {
		case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', LNAME, '@', '?', '(':
			// sym name_or_type
			typ := p.ntype()
			nn := Nod(ONONAME, nil, nil)
			nn.Sym = name
			return Nod(OKEY, nn, typ)

		case LDDD:
			// sym dotdotdot
			typ := p.dotdotdot()
			nn := Nod(ONONAME, nil, nil)
			nn.Sym = name
			return Nod(OKEY, nn, typ)

		default:
			// name_or_type
			name := mkname(name)
			// from dotname
			if p.got('.') {
				return p.new_dotname(name)
			}
			return name
		}

	case LDDD:
		// dotdotdot
		return p.dotdotdot()

	case LCOMM, LFUNC, '[', LCHAN, LMAP, LSTRUCT, LINTERFACE, '*', '(':
		// name_or_type
		return p.ntype()

	default:
		p.syntax_error("expecting )")
		p.advance(',', ')')
		return nil
	}
}

// Parameters    = "(" [ ParameterList [ "," ] ] ")" .
// ParameterList = ParameterDecl { "," ParameterDecl } .
func (p *parser) param_list() (l *NodeList) {
	if trace && Debug['x'] != 0 {
		defer p.trace("param_list")()
	}

	p.want('(')

	for p.tok != EOF && p.tok != ')' {
		l = list(l, p.arg_type())
		if !p.ocomma(')') {
			break
		}
	}

	p.want(')')
	return
}

var missing_stmt = Nod(OXXX, nil, nil)

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
		return p.compound_stmt(false)

	case LVAR, LCONST, LTYPE:
		return liststmt(p.common_dcl())

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
		stmt := Nod(OXFALL, nil, nil)
		stmt.Xoffset = int64(block)
		return stmt

	case LBREAK:
		p.next()
		return Nod(OBREAK, p.onew_name(), nil)

	case LCONTINUE:
		p.next()
		return Nod(OCONTINUE, p.onew_name(), nil)

	case LGO:
		p.next()
		return Nod(OPROC, p.pseudocall(), nil)

	case LDEFER:
		p.next()
		return Nod(ODEFER, p.pseudocall(), nil)

	case LGOTO:
		p.next()
		stmt := Nod(OGOTO, p.new_name(p.sym()), nil)
		stmt.Sym = dclstack // context, for goto restrictions
		return stmt

	case LRETURN:
		p.next()
		var results *NodeList
		if p.tok != ';' && p.tok != '}' {
			results = p.expr_list()
		}

		stmt := Nod(ORETURN, nil, nil)
		stmt.List = results
		if stmt.List == nil && Curfn != nil {
			for l := Curfn.Func.Dcl; l != nil; l = l.Next {
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

		return stmt

	case ';':
		return nil

	default:
		return missing_stmt
	}
}

// StatementList = { Statement ";" } .
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
			p.syntax_error("at end of statement")
			p.advance(';', '}')
		}
	}
	return
}

// IdentifierList = identifier { "," identifier } .
//
// If first != nil we have the first symbol already.
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

// IdentifierList = identifier { "," identifier } .
func (p *parser) dcl_name_list() *NodeList {
	if trace && Debug['x'] != 0 {
		defer p.trace("dcl_name_list")()
	}

	l := list1(p.dcl_name(p.sym()))
	for p.got(',') {
		l = list(l, p.dcl_name(p.sym()))
	}
	return l
}

// ExpressionList = Expression { "," Expression } .
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

// Arguments = "(" [ ( ExpressionList | Type [ "," ExpressionList ] ) [ "..." ] [ "," ] ] ")" .
func (p *parser) arg_list() (l *NodeList, ddd bool) {
	if trace && Debug['x'] != 0 {
		defer p.trace("arg_list")()
	}

	p.want('(')
	p.xnest++

	for p.tok != EOF && p.tok != ')' && !ddd {
		l = list(l, p.expr()) // expr_or_type
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

// ----------------------------------------------------------------------------
// Importing packages

func (p *parser) import_error() {
	p.syntax_error("in export data of imported package")
	p.next()
}

// The methods below reflect a 1:1 translation of the original (and now defunct)
// go.y yacc productions. They could be simplified significantly and also use better
// variable names. However, we will be able to delete them once we enable the
// new export format by default, so it's not worth the effort (issue 13241).

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

func (p *parser) hidden_pkg_importsym() *Sym {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_pkg_importsym")()
	}

	s1 := p.hidden_importsym()

	ss := s1
	structpkg = ss.Pkg

	return ss
}

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

func (p *parser) hidden_interfacedcl() *Node {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_interfacedcl")()
	}

	// The original (now defunct) grammar in go.y accepted both a method
	// or an (embedded) type:
	//
	// hidden_interfacedcl:
	// 	sym '(' ohidden_funarg_list ')' ohidden_funres
	// 	{
	// 		$$ = Nod(ODCLFIELD, newname($1), typenod(functype(fakethis(), $3, $5)));
	// 	}
	// |	hidden_type
	// 	{
	// 		$$ = Nod(ODCLFIELD, nil, typenod($1));
	// 	}
	//
	// But the current textual export code only exports (inlined) methods,
	// even if the methods came from embedded interfaces. Furthermore, in
	// the original grammar, hidden_type may also start with a sym (LNAME
	// or '@'), complicating matters further. Since we never have embedded
	// types, only parse methods here.

	s1 := p.sym()
	p.want('(')
	s3 := p.ohidden_funarg_list()
	p.want(')')
	s5 := p.ohidden_funres()

	return Nod(ODCLFIELD, newname(s1), typenod(functype(fakethis(), s3, s5)))
}

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

func (p *parser) hidden_import_list() {
	if trace && Debug['x'] != 0 {
		defer p.trace("hidden_import_list")()
	}

	for p.tok != '$' {
		p.hidden_import()
	}
}

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
