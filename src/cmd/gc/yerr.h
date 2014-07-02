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

	{222, ',',
	"unexpected comma during import block"},

	{32, ';',
	"missing import path; require quoted string"},

	{378, ';',
	"missing { after if clause"},

	{399, ';',
	"missing { after switch clause"},

	{238, ';',
	"missing { after for clause"},

	{476, LBODY,
	"missing { after for clause"},

	{22, '{',
	"unexpected semicolon or newline before {"},

	{145, ';',
	"unexpected semicolon or newline in type declaration"},

	{37, '}',
	"unexpected } in channel type"},
	
	{37, ')',
	"unexpected ) in channel type"},
	
	{37, ',',
	"unexpected comma in channel type"},

	{439, LELSE,
	"unexpected semicolon or newline before else"},

	{258, ',',
	"name list not allowed in interface type"},

	{238, LVAR,
	"var declaration not allowed in for initializer"},

	{65, '{',
	"unexpected { at end of statement"},

	{377, '{',
	"unexpected { at end of statement"},
	
	{126, ';',
	"argument to go/defer must be function call"},
	
	{426, ';',
	"need trailing comma before newline in composite literal"},
	
	{437, ';',
	"need trailing comma before newline in composite literal"},
	
	{113, LNAME,
	"nested func not allowed"},

	{645, ';',
	"else must be followed by if or statement block"}
};
