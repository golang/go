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
	"flag"
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

func ExampleKeyValue() {
	v := struct {
		a string
		b int
	}{
		a: "A",
		b: 1,
	}
	fmt.Print(v)
	// Output: a: "A", b: 1
}

func ExampleKeyValueImport() {
	f := flag.Flag{
		Name: "play",
	}
	fmt.Print(f)
	// Output: Name: "play"
}

var keyValueTopDecl = struct {
	a string
	b int
}{
	a: "B",
	b: 2,
}

func ExampleKeyValueTopDecl() {
	fmt.Print(keyValueTopDecl)
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
	{
		Name:   "KeyValue",
		Play:   exampleKeyValuePlay,
		Output: "a: \"A\", b: 1\n",
	},
	{
		Name:   "KeyValueImport",
		Play:   exampleKeyValueImportPlay,
		Output: "Name: \"play\"\n",
	},
	{
		Name: "KeyValueTopDecl",
		Play: "<nil>",
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

const exampleKeyValuePlay = `package main

import (
	"fmt"
)

func main() {
	v := struct {
		a string
		b int
	}{
		a: "A",
		b: 1,
	}
	fmt.Print(v)
}
`

const exampleKeyValueImportPlay = `package main

import (
	"flag"
	"fmt"
)

func main() {
	f := flag.Flag{
		Name: "play",
	}
	fmt.Print(f)
}
`

func TestExamples(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", strings.NewReader(exampleTestFile), parser.ParseComments)
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
				var buf bytes.Buffer
				if err := format.Node(&buf, fset, e.Play); err != nil {
					t.Fatal(err)
				}
				g = buf.String()
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
