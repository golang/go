// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"template"
	"testing"
)

type data struct {
	F, T    bool
	C, G, H string
	A, E    []string
}

var testData = data{
	F: false,
	T: true,
	C: "<Cincinatti>",
	G: "<Goodbye>",
	H: "<Hello>",
	A: []string{"<a>", "<b>"},
	E: []string{},
}

type testCase struct {
	name   string
	input  string
	output string
}

var testCases = []testCase{
	{"if", "{{if .T}}Hello{{end}}, {{.C}}!", "Hello, &lt;Cincinatti&gt;!"},
	{"else", "{{if .F}}{{.H}}{{else}}{{.G}}{{end}}!", "&lt;Goodbye&gt;!"},
	{"overescaping", "Hello, {{.C | html}}!", "Hello, &lt;Cincinatti&gt;!"},
	{"assignment", "{{if $x := .H}}{{$x}}{{end}}", "&lt;Hello&gt;"},
	{"withBody", "{{with .H}}{{.}}{{end}}", "&lt;Hello&gt;"},
	{"withElse", "{{with .E}}{{.}}{{else}}{{.H}}{{end}}", "&lt;Hello&gt;"},
	{"rangeBody", "{{range .A}}{{.}}{{end}}", "&lt;a&gt;&lt;b&gt;"},
	{"rangeElse", "{{range .E}}{{.}}{{else}}{{.H}}{{end}}", "&lt;Hello&gt;"},
	{"nonStringValue", "{{.T}}", "true"},
	{"constant", `<a href="{{"'str'"}}">`, `<a href="&#39;str&#39;">`},
}

func TestAutoesc(t *testing.T) {
	for _, testCase := range testCases {
		name := testCase.name
		tmpl := template.New(name)
		tmpl, err := tmpl.Parse(testCase.input)
		if err != nil {
			t.Errorf("%s: failed to parse template: %s", name, err)
			continue
		}

		Escape(tmpl)

		buffer := new(bytes.Buffer)

		err = tmpl.Execute(buffer, testData)
		if err != nil {
			t.Errorf("%s: template execution failed: %s", name, err)
			continue
		}

		output := testCase.output
		actual := buffer.String()
		if output != actual {
			t.Errorf("%s: escaped output: %q != %q",
				name, output, actual)
		}
	}
}
