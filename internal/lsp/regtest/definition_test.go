// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"path"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
)

const internalDefinition = `
-- go.mod --
module mod

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println(message)
}
-- const.go --
package main

const message = "Hello World."
`

func TestGoToInternalDefinition(t *testing.T) {
	t.Parallel()
	runner.Run(t, internalDefinition, func(ctx context.Context, t *testing.T, env *Env) {
		env.OpenFile("main.go")
		name, pos := env.GoToDefinition("main.go", fake.Pos{Line: 5, Column: 13})
		if want := "const.go"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
		if want := (fake.Pos{Line: 2, Column: 6}); pos != want {
			t.Errorf("GoToDefinition: got position %v, want %v", pos, want)
		}
	})
}

const stdlibDefinition = `
-- go.mod --
module mod

go 1.12
-- main.go --
package main

import (
	"fmt"
	"time"
)

func main() {
	fmt.Println(time.Now())
}`

func TestGoToStdlibDefinition(t *testing.T) {
	t.Parallel()
	runner.Run(t, stdlibDefinition, func(ctx context.Context, t *testing.T, env *Env) {
		env.OpenFile("main.go")
		name, pos := env.GoToDefinition("main.go", fake.Pos{Line: 8, Column: 19})
		if got, want := path.Base(name), "time.go"; got != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}

		// Test that we can jump to definition from outside our workspace.
		// See golang.org/issues/37045.
		newName, newPos := env.GoToDefinition(name, pos)
		if newName != name {
			t.Errorf("GoToDefinition is not idempotent: got %q, want %q", newName, name)
		}
		if newPos != pos {
			t.Errorf("GoToDefinition is not idempotent: got %v, want %v", newPos, pos)
		}
	})
}
