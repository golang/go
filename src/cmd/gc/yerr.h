// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Example-based syntax error messages.
// See bisonerrors, Makefile, go.y.

static struct {
	int yystate;
	int yychar;
	char *msg;
} yymsg[] = {
	// Each line of the form % token list
	// is converted by bisonerrors into the yystate and yychar caused
	// by that token list.

	220, ',',
	"unexpected comma during import block",

	376, ';',
	"unexpected semicolon or newline before {",

	397, ';',
	"unexpected semicolon or newline before {",

	236, ';',
	"unexpected semicolon or newline before {",

	473, LBODY,
	"unexpected semicolon or newline before {",

	22, '{',
	"unexpected semicolon or newline before {",

	143, ';',
	"unexpected semicolon or newline in type declaration",

	37, '}',
	"unexpected } in channel type",
	
	37, ')',
	"unexpected ) in channel type",
	
	37, ',',
	"unexpected comma in channel type",

	436, LELSE,
	"unexpected semicolon or newline before else",

	256, ',',
	"name list not allowed in interface type",

	236, LVAR,
	"var declaration not allowed in for initializer",

	65, '{',
	"unexpected { at end of statement",

	375, '{',
	"unexpected { at end of statement",
	
	124, ';',
	"argument to go/defer must be function call",
	
	424, ';',
	"need trailing comma before newline in composite literal",
	
	111, LNAME,
	"nested func not allowed",

	614, ';',
	"else must be followed by if or statement block"
};
