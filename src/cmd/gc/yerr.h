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

	221, ',',
	"unexpected comma during import block",

	377, ';',
	"unexpected semicolon or newline before {",

	398, ';',
	"unexpected semicolon or newline before {",

	237, ';',
	"unexpected semicolon or newline before {",

	475, LBODY,
	"unexpected semicolon or newline before {",

	22, '{',
	"unexpected semicolon or newline before {",

	144, ';',
	"unexpected semicolon or newline in type declaration",

	37, '}',
	"unexpected } in channel type",
	
	37, ')',
	"unexpected ) in channel type",
	
	37, ',',
	"unexpected comma in channel type",

	438, LELSE,
	"unexpected semicolon or newline before else",

	257, ',',
	"name list not allowed in interface type",

	237, LVAR,
	"var declaration not allowed in for initializer",

	65, '{',
	"unexpected { at end of statement",

	376, '{',
	"unexpected { at end of statement",
	
	125, ';',
	"argument to go/defer must be function call",
	
	425, ';',
	"need trailing comma before newline in composite literal",
	
	436, ';',
	"need trailing comma before newline in composite literal",
	
	112, LNAME,
	"nested func not allowed",

	642, ';',
	"else must be followed by if or statement block"
};
