// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"path"
	"testing"
)

const internalDefinition = `
-- go.mod --
module mod.com

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
	runner.Run(t, internalDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		name, pos := env.GoToDefinition("main.go", env.RegexpSearch("main.go", "message"))
		if want := "const.go"; name != want {
			t.Errorf("GoToDefinition: got file %q, want %q", name, want)
		}
		if want := env.RegexpSearch("const.go", "message"); pos != want {
			t.Errorf("GoToDefinition: got position %v, want %v", pos, want)
		}
	})
}

const stdlibDefinition = `
-- go.mod --
module mod.com

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

func TestGoToStdlibDefinition_Issue37045(t *testing.T) {
	runner.Run(t, stdlibDefinition, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		name, pos := env.GoToDefinition("main.go", env.RegexpSearch("main.go", "Now"))
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
