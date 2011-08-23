// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"strings"
	"template"
	"template/parse"
	"testing"
)

func TestEscape(t *testing.T) {
	var data = struct {
		F, T    bool
		C, G, H string
		A, E    []string
	}{
		F: false,
		T: true,
		C: "<Cincinatti>",
		G: "<Goodbye>",
		H: "<Hello>",
		A: []string{"<a>", "<b>"},
		E: []string{},
	}

	var testCases = []struct {
		name   string
		input  string
		output string
	}{
		{
			"if",
			"{{if .T}}Hello{{end}}, {{.C}}!",
			"Hello, &lt;Cincinatti&gt;!",
		},
		{
			"else",
			"{{if .F}}{{.H}}{{else}}{{.G}}{{end}}!",
			"&lt;Goodbye&gt;!",
		},
		{
			"overescaping",
			"Hello, {{.C | html}}!",
			"Hello, &lt;Cincinatti&gt;!",
		},
		{
			"assignment",
			"{{if $x := .H}}{{$x}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"withBody",
			"{{with .H}}{{.}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"withElse",
			"{{with .E}}{{.}}{{else}}{{.H}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"rangeBody",
			"{{range .A}}{{.}}{{end}}",
			"&lt;a&gt;&lt;b&gt;",
		},
		{
			"rangeElse",
			"{{range .E}}{{.}}{{else}}{{.H}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"nonStringValue",
			"{{.T}}",
			"true",
		},
		{
			// TODO: Make sure the URL escaper escapes single quotes so it can
			// be embedded in single quoted URI attributes and CSS url(...)
			// constructs. Single quotes are reserved in URLs, but are only used
			// in the obsolete "mark" rule in an appendix in RFC 3986 so can be
			// safely encoded.
			"constant",
			`<a href="{{"'a<b'"}}">`,
			`<a href="'a%3Cb'">`,
		},
	}

	for _, tc := range testCases {
		tmpl, err := template.New(tc.name).Parse(tc.input)
		if err != nil {
			t.Errorf("%s: template parsing failed: %s", tc.name, err)
			continue
		}
		Escape(tmpl)
		b := new(bytes.Buffer)
		if err = tmpl.Execute(b, data); err != nil {
			t.Errorf("%s: template execution failed: %s", tc.name, err)
			continue
		}
		if w, g := tc.output, b.String(); w != g {
			t.Errorf("%s: escaped output: want %q got %q", tc.name, w, g)
			continue
		}
	}
}

func TestErrors(t *testing.T) {
	var testCases = []struct {
		input string
		err   string
	}{
		// Non-error cases.
		{
			"{{if .Cond}}<a>{{else}}<b>{{end}}",
			"",
		},
		{
			"{{if .Cond}}<a>{{end}}",
			"",
		},
		{
			"{{if .Cond}}{{else}}<b>{{end}}",
			"",
		},
		{
			"{{with .Cond}}<div>{{end}}",
			"",
		},
		{
			"{{range .Items}}<a>{{end}}",
			"",
		},
		{
			"<a href='/foo?{{range .Items}}&{{.K}}={{.V}}{{end}}'>",
			"",
		},
		// Error cases.
		{
			"{{if .Cond}}<a{{end}}",
			"z:1: {{if}} branches",
		},
		{
			"{{if .Cond}}\n{{else}}\n<a{{end}}",
			"z:1: {{if}} branches",
		},
		/*
			TODO: Should the error really be non-empty? Both branches close the tag...

			// Missing quote in the else branch.
			{
				`{{if .Cond}}<a href="foo">{{else}}<a href="bar>{{end}}`,
				"z:1: {{if}} branches",
			},
		*/
		{
			// Different kind of attribute: href implies a URL.
			"<a {{if .Cond}}href='{{else}}title='{{end}}{{.X}}'>",
			"z:1: {{if}} branches",
		},
		{
			"\n{{with .X}}<a{{end}}",
			"z:2: {{with}} branches",
		},
		{
			"\n{{with .X}}<a>{{else}}<a{{end}}",
			"z:2: {{with}} branches",
		},
		{
			"{{range .Items}}<a{{end}}",
			"z:1: {{range}} branches",
		},
		{
			"\n{{range .Items}} x='<a{{end}}",
			"z:2: {{range}} branches",
		},
	}

	for _, tc := range testCases {
		tmpl, err := template.New("z").Parse(tc.input)
		if err != nil {
			t.Errorf("input=%q: template parsing failed: %s", tc.input, err)
			continue
		}
		var got string
		if _, err := Escape(tmpl); err != nil {
			got = err.String()
		}
		if tc.err == "" {
			if got != "" {
				t.Errorf("input=%q: unexpected error %q", tc.input, got)
			}
			continue
		}
		if strings.Index(got, tc.err) == -1 {
			t.Errorf("input=%q: error %q does not contain expected string %q", tc.input, got, tc.err)
			continue
		}
	}
}

func TestEscapeText(t *testing.T) {
	var testCases = []struct {
		input  string
		output context
	}{
		{
			``,
			context{},
		},
		{
			`Hello, World!`,
			context{},
		},
		{
			// An orphaned "<" is OK.
			`I <3 Ponies!`,
			context{},
		},
		{
			`<a`,
			context{state: stateTag},
		},
		{
			`<a `,
			context{state: stateTag},
		},
		{
			`<a>`,
			context{state: stateText},
		},
		{
			`<a href=`,
			context{state: stateURL, delim: delimSpaceOrTagEnd},
		},
		{
			`<a href ='`,
			context{state: stateURL, delim: delimSingleQuote},
		},
		{
			`<a href= "`,
			context{state: stateURL, delim: delimDoubleQuote},
		},
		{
			`<a title="`,
			context{state: stateAttr, delim: delimDoubleQuote},
		},
		{
			`<a HREF='http:`,
			context{state: stateURL, delim: delimSingleQuote},
		},
		{
			`<a Href='/`,
			context{state: stateURL, delim: delimSingleQuote},
		},
	}

	for _, tc := range testCases {
		n := &parse.TextNode{
			NodeType: parse.NodeText,
			Text:     []byte(tc.input),
		}
		c := escapeText(context{}, n)
		if !tc.output.eq(c) {
			t.Errorf("input %q: want context %v got %v", tc.input, tc.output, c)
			continue
		}
		if tc.input != string(n.Text) {
			t.Errorf("input %q: text node was modified: want %q got %q", tc.input, tc.input, n.Text)
			continue
		}
	}
}
