// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/protocol"
	. "golang.org/x/tools/internal/lsp/regtest"
	"golang.org/x/tools/internal/lsp/tests"
)

func TestAddImport(t *testing.T) {
	const before = `package main

import "fmt"

func main() {
	fmt.Println("hello world")
}
`

	const want = `package main

import (
	"bytes"
	"fmt"
)

func main() {
	fmt.Println("hello world")
}
`

	Run(t, "", func(t *testing.T, env *Env) {
		env.CreateBuffer("main.go", before)
		cmd, err := command.NewAddImportCommand("Add Import", command.AddImportArgs{
			URI:        env.Sandbox.Workdir.URI("main.go"),
			ImportPath: "bytes",
		})
		if err != nil {
			t.Fatal(err)
		}
		env.ExecuteCommand(&protocol.ExecuteCommandParams{
			Command:   "gopls.add_import",
			Arguments: cmd.Arguments,
		}, nil)
		got := env.Editor.BufferText("main.go")
		if got != want {
			t.Fatalf("gopls.add_import failed\n%s", tests.Diff(t, want, got))
		}
	})
}

func TestListImports(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- foo.go --
package foo
const C = 1
-- import_strings_test.go --
package foo
import (
	x "strings"
	"testing"
)

func TestFoo(t *testing.T) {}
-- import_testing_test.go --
package foo

import "testing"

func TestFoo2(t *testing.T) {}
`
	tests := []struct {
		filename string
		want     command.ListImportsResult
	}{
		{
			filename: "import_strings_test.go",
			want: command.ListImportsResult{
				Imports: []command.FileImport{
					{Name: "x", Path: "strings"},
					{Path: "testing"},
				},
				PackageImports: []command.PackageImport{
					{Path: "strings"},
					{Path: "testing"},
				},
			},
		},
		{
			filename: "import_testing_test.go",
			want: command.ListImportsResult{
				Imports: []command.FileImport{
					{Path: "testing"},
				},
				PackageImports: []command.PackageImport{
					{Path: "strings"},
					{Path: "testing"},
				},
			},
		},
	}

	Run(t, files, func(t *testing.T, env *Env) {
		for _, tt := range tests {
			cmd, err := command.NewListImportsCommand("List Imports", command.URIArg{
				URI: env.Sandbox.Workdir.URI(tt.filename),
			})
			if err != nil {
				t.Fatal(err)
			}
			var result command.ListImportsResult
			env.ExecuteCommand(&protocol.ExecuteCommandParams{
				Command:   command.ListImports.ID(),
				Arguments: cmd.Arguments,
			}, &result)
			if diff := cmp.Diff(tt.want, result); diff != "" {
				t.Errorf("unexpected list imports result for %q (-want +got):\n%s", tt.filename, diff)
			}
		}

	})
}
