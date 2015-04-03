// DO NOT EDIT - generated with go generate

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Example-based syntax error messages.
// See yaccerrors.go.

package gc

var yymsg = []struct {
	yystate int
	yychar  int
	msg     string
}{
	// Each line of the form % token list
	// is converted by yaccerrors.go into the yystate and yychar caused
	// by that token list.

	{332, ',',
		"unexpected comma during import block"},

	{89, ';',
		"missing import path; require quoted string"},

	{390, ';',
		"missing { after if clause"},

	{387, ';',
		"missing { after switch clause"},

	{279, ';',
		"missing { after for clause"},

	{498, LBODY,
		"missing { after for clause"},

	{17, '{',
		"unexpected semicolon or newline before {"},

	{111, ';',
		"unexpected semicolon or newline in type declaration"},

	{78, '}',
		"unexpected } in channel type"},

	{78, ')',
		"unexpected ) in channel type"},

	{78, ',',
		"unexpected comma in channel type"},

	{416, LELSE,
		"unexpected semicolon or newline before else"},

	{329, ',',
		"name list not allowed in interface type"},

	{279, LVAR,
		"var declaration not allowed in for initializer"},

	{25, '{',
		"unexpected { at end of statement"},

	{371, '{',
		"unexpected { at end of statement"},

	{122, ';',
		"argument to go/defer must be function call"},

	{398, ';',
		"need trailing comma before newline in composite literal"},

	{414, ';',
		"need trailing comma before newline in composite literal"},

	{124, LNAME,
		"nested func not allowed"},

	{650, ';',
		"else must be followed by if or statement block"},
}
