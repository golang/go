// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template_test

import (
	"log"
	"os"
	"strings"
	"text/template"
)

// This example demonstrates a custom function to process template text.
// It installs the strings.Title function and uses it to
// Make Title Text Look Good In Our Template's Output.
func ExampleTemplate_func() {
	// First we create a FuncMap with which to register the function.
	funcMap := template.FuncMap{
		// The name "title" is what the function will be called in the template text.
		"title": strings.Title,
	}

	// A simple template definition to test our function.
	// We print the input text several ways:
	// - the original
	// - title-cased
	// - title-cased and then printed with %q
	// - printed with %q and then title-cased.
	const templateText = `
Input: {{printf "%q" .}}
Output 0: {{title .}}
Output 1: {{title . | printf "%q"}}
Output 2: {{printf "%q" . | title}}
`

	// Create a template, add the function map, and parse the text.
	tmpl, err := template.New("titleTest").Funcs(funcMap).Parse(templateText)
	if err != nil {
		log.Fatalf("parsing: %s", err)
	}

	// Run the template to verify the output.
	err = tmpl.Execute(os.Stdout, "the go programming language")
	if err != nil {
		log.Fatalf("execution: %s", err)
	}

	// Output:
	// Input: "the go programming language"
	// Output 0: The Go Programming Language
	// Output 1: "The Go Programming Language"
	// Output 2: "The Go Programming Language"
}

// This example demonstrates how to use 'gt' with two parameters with 'if/else'.
func ExampleTemplate_func() {
	type book struct {
		Stars float32
		Name  string
	}
	tpl, err := template.New("book").Parse(`{{ if (gt .Stars 4.0) }}'{{.Name }}' is a great book.{{ else }} '{{.Name}}' is not so good.{{ end }}`)
	if err != nil {
		log.Fatalf("failed to parse template: %s", err)
	}

	b := &book{
		Stars: 4.5,
		Name:  "Good Night, Gopher",
	}
	err = tpl.Execute(os.Stdout, b)
	if err != nil {
		log.Fatalf("failed to execute template: %s", err)
	}
	// Output:
	// 'Good Night, Gopher' is a great book.
}
