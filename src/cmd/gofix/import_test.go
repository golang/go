// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast"

func init() {
	addTestCases(importTests, nil)
}

var importTests = []testCase{
	{
		Name: "import.0",
		Fn:   addImportFn("os"),
		In: `package main

import (
	"os"
)
`,
		Out: `package main

import (
	"os"
)
`,
	},
	{
		Name: "import.1",
		Fn:   addImportFn("os"),
		In: `package main
`,
		Out: `package main

import "os"
`,
	},
	{
		Name: "import.2",
		Fn:   addImportFn("os"),
		In: `package main

// Comment
import "C"
`,
		Out: `package main

// Comment
import "C"
import "os"
`,
	},
	{
		Name: "import.3",
		Fn:   addImportFn("os"),
		In: `package main

// Comment
import "C"

import (
	"io"
	"utf8"
)
`,
		Out: `package main

// Comment
import "C"

import (
	"io"
	"os"
	"utf8"
)
`,
	},
	{
		Name: "import.4",
		Fn:   deleteImportFn("os"),
		In: `package main

import (
	"os"
)
`,
		Out: `package main
`,
	},
	{
		Name: "import.5",
		Fn:   deleteImportFn("os"),
		In: `package main

// Comment
import "C"
import "os"
`,
		Out: `package main

// Comment
import "C"
`,
	},
	{
		Name: "import.6",
		Fn:   deleteImportFn("os"),
		In: `package main

// Comment
import "C"

import (
	"io"
	"os"
	"utf8"
)
`,
		Out: `package main

// Comment
import "C"

import (
	"io"
	"utf8"
)
`,
	},
	{
		Name: "import.7",
		Fn:   deleteImportFn("io"),
		In: `package main

import (
	"io"   // a
	"os"   // b
	"utf8" // c
)
`,
		Out: `package main

import (
	// a
	"os"   // b
	"utf8" // c
)
`,
	},
	{
		Name: "import.8",
		Fn:   deleteImportFn("os"),
		In: `package main

import (
	"io"   // a
	"os"   // b
	"utf8" // c
)
`,
		Out: `package main

import (
	"io" // a
	// b
	"utf8" // c
)
`,
	},
	{
		Name: "import.9",
		Fn:   deleteImportFn("utf8"),
		In: `package main

import (
	"io"   // a
	"os"   // b
	"utf8" // c
)
`,
		Out: `package main

import (
	"io" // a
	"os" // b
	// c
)
`,
	},
	{
		Name: "import.10",
		Fn:   deleteImportFn("io"),
		In: `package main

import (
	"io"
	"os"
	"utf8"
)
`,
		Out: `package main

import (
	"os"
	"utf8"
)
`,
	},
	{
		Name: "import.11",
		Fn:   deleteImportFn("os"),
		In: `package main

import (
	"io"
	"os"
	"utf8"
)
`,
		Out: `package main

import (
	"io"
	"utf8"
)
`,
	},
	{
		Name: "import.12",
		Fn:   deleteImportFn("utf8"),
		In: `package main

import (
	"io"
	"os"
	"utf8"
)
`,
		Out: `package main

import (
	"io"
	"os"
)
`,
	},
	{
		Name: "import.13",
		Fn:   rewriteImportFn("utf8", "encoding/utf8"),
		In: `package main

import (
	"io"
	"os"
	"utf8" // thanks ken
)
`,
		Out: `package main

import (
	"encoding/utf8" // thanks ken
	"io"
	"os"
)
`,
	},
}

func addImportFn(path string) func(*ast.File) bool {
	return func(f *ast.File) bool {
		if !imports(f, path) {
			addImport(f, path)
			return true
		}
		return false
	}
}

func deleteImportFn(path string) func(*ast.File) bool {
	return func(f *ast.File) bool {
		if imports(f, path) {
			deleteImport(f, path)
			return true
		}
		return false
	}
}

func rewriteImportFn(old, new string) func(*ast.File) bool {
	return func(f *ast.File) bool {
		if imports(f, old) {
			rewriteImport(f, old, new)
			return true
		}
		return false
	}
}
