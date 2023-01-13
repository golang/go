// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

const exampleProgram = `
-- go.mod --
go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}
`

func TestClientEditing(t *testing.T) {
	ws, err := NewSandbox(&SandboxConfig{Files: UnpackTxt(exampleProgram)})
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	ctx := context.Background()
	editor := NewEditor(ws, EditorConfig{})
	if err := editor.OpenFile(ctx, "main.go"); err != nil {
		t.Fatal(err)
	}
	if err := editor.EditBuffer(ctx, "main.go", []protocol.TextEdit{
		{
			Range: protocol.Range{
				Start: protocol.Position{Line: 5, Character: 14},
				End:   protocol.Position{Line: 5, Character: 26},
			},
			NewText: "Hola, mundo.",
		},
	}); err != nil {
		t.Fatal(err)
	}
	got := editor.buffers["main.go"].text()
	want := `package main

import "fmt"

func main() {
	fmt.Println("Hola, mundo.")
}
`
	if got != want {
		t.Errorf("got text %q, want %q", got, want)
	}
}
