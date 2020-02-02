// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"testing"
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
	ws, err := NewWorkspace("test", []byte(exampleProgram))
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	ctx := context.Background()
	client := NewEditor(ws)
	if err != nil {
		t.Fatal(err)
	}
	if err := client.OpenFile(ctx, "main.go"); err != nil {
		t.Fatal(err)
	}
	if err := client.EditBuffer(ctx, "main.go", []Edit{
		{
			Start: Pos{5, 14},
			End:   Pos{5, 26},
			Text:  "Hola, mundo.",
		},
	}); err != nil {
		t.Fatal(err)
	}
	got := client.buffers["main.go"].text()
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
