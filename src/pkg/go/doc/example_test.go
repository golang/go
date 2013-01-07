// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc_test

import (
	"bytes"
	"go/doc"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

const exampleTestFile = `
package foo_test

import (
	"fmt"
	"log"
	"os/exec"
)

func ExampleHello() {
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}

func ExampleImport() {
	out, err := exec.Command("date").Output()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("The date is %s\n", out)
}
`

var exampleTestCases = []struct {
	Name, Play, Output string
}{
	{
		Name:   "Hello",
		Play:   exampleHelloPlay,
		Output: "Hello, world!\n",
	},
	{
		Name: "Import",
		Play: exampleImportPlay,
	},
}

const exampleHelloPlay = `package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello, world!")
}
`
const exampleImportPlay = `package main

import (
	"fmt"
	"log"
	"os/exec"
)

func main() {
	out, err := exec.Command("date").Output()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("The date is %s\n", out)
}
`

func TestExamples(t *testing.T) {
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "test.go", strings.NewReader(exampleTestFile), parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	for i, e := range doc.Examples(file) {
		c := exampleTestCases[i]
		if e.Name != c.Name {
			t.Errorf("got Name == %q, want %q", e.Name, c.Name)
		}
		if w := c.Play; w != "" {
			var g string // hah
			if e.Play == nil {
				g = "<nil>"
			} else {
				b := new(bytes.Buffer)
				if err := format.Node(b, fs, e.Play); err != nil {
					t.Fatal(err)
				}
				g = b.String()
			}
			if g != w {
				t.Errorf("%s: got Play == %q, want %q", c.Name, g, w)
			}
		}
		if g, w := e.Output, c.Output; g != w {
			t.Errorf("%s: got Output == %q, want %q", c.Name, g, w)
		}
	}
}
