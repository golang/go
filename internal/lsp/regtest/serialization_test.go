// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"encoding/json"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
)

const simpleProgram = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}`

func TestHoverSerialization(t *testing.T) {
	runner.Run(t, simpleProgram, func(t *testing.T, env *Env) {
		// Hover on an empty line.
		params := protocol.HoverParams{}
		params.TextDocument.URI = env.Sandbox.Workdir.URI("main.go")
		params.Position.Line = 3
		params.Position.Character = 0
		var resp json.RawMessage
		if err := protocol.Call(env.Ctx, env.Conn, "textDocument/hover", &params, &resp); err != nil {
			t.Fatal(err)
		}
		if len(string(resp)) > 0 {
			t.Errorf("got non-empty response for empty hover: %v", string(resp))
		}
	})
}
