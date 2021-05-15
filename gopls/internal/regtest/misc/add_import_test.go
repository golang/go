// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

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
			URI:        protocol.URIFromSpanURI(env.Sandbox.Workdir.URI("main.go").SpanURI()),
			ImportPath: "bytes",
		})
		if err != nil {
			t.Fatal(err)
		}
		_, err = env.Editor.ExecuteCommand(env.Ctx, &protocol.ExecuteCommandParams{
			Command:   "gopls.add_import",
			Arguments: cmd.Arguments,
		})
		if err != nil {
			t.Fatal(err)
		}
		got := env.Editor.BufferText("main.go")
		if got != want {
			t.Fatalf("gopls.add_import failed\n%s", tests.Diff(t, want, got))
		}
	})
}
