// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
)

const simpleProgram = `
-- go.mod --
module gopls.test

go 1.12
-- lib/lib.go --
package lib

const Hello = "Hello"
-- main.go --
package main

import (
	"fmt"
	"gopls.test/lib"
)

func main() {
	fmt.Println(lib.Hello)
}`

func TestHover(t *testing.T) {
	runner.Run(t, simpleProgram, func(t *testing.T, env *Env) {
		// Hover on an empty line.
		env.OpenFile("main.go")
		content, pos := env.Hover("main.go", fake.Pos{Line: 3, Column: 0})
		if content != nil {
			t.Errorf("got non-empty response for empty hover: %v: %v", pos, *content)
		}
		content, pos = env.Hover("main.go", env.RegexpSearch("main.go", "lib.Hello"))
		link := "pkg.go.dev/gopls.test/lib"
		if content == nil || !strings.Contains(content.Value, link) {
			t.Errorf("got hover: %v, want contains %q", content, link)
		}
		env.ChangeEnv("GOPRIVATE=gopls.test")
		content, pos = env.Hover("main.go", env.RegexpSearch("main.go", "lib.Hello"))
		if content == nil || strings.Contains(content.Value, link) {
			t.Errorf("got hover: %v, want non-empty hover without %q", content, link)
		}
	})
}
