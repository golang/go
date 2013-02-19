// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Yacc is a version of yacc for Go.
It is written in Go and generates parsers written in Go.

Usage:

	go tool yacc args...

It is largely transliterated from the Inferno version written in Limbo
which in turn was largely transliterated from the Plan 9 version
written in C and documented at

	http://plan9.bell-labs.com/magic/man2html/1/yacc

Adepts of the original yacc will have no trouble adapting to this
form of the tool.

The file units.y in this directory is a yacc grammar for a version of
the Unix tool units, also written in Go and largely transliterated
from the Plan 9 C version. It needs the flag "-p units_" (see
below).

The generated parser is reentrant. Parse expects to be given an
argument that conforms to the following interface:

	type yyLexer interface {
		Lex(lval *yySymType) int
		Error(e string)
	}

Lex should return the token identifier, and place other token
information in lval (which replaces the usual yylval).
Error is equivalent to yyerror in the original yacc.

Code inside the parser may refer to the variable yylex,
which holds the yyLexer passed to Parse.

Multiple grammars compiled into a single program should be placed in
distinct packages.  If that is impossible, the "-p prefix" flag to
yacc sets the prefix, by default yy, that begins the names of
symbols, including types, the parser, and the lexer, generated and
referenced by yacc's generated code.  Setting it to distinct values
allows multiple grammars to be placed in a single package.

*/
package main
