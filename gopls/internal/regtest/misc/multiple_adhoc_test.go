// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestMultipleAdHocPackages(t *testing.T) {
	Run(t, `
-- a/a.go --
package main

import "fmt"

func main() {
	fmt.Println("")
}
-- a/b.go --
package main

import "fmt"

func main() () {
	fmt.Println("")
}
`, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		if list := env.Completion("a/a.go", env.RegexpSearch("a/a.go", "Println")); list == nil || len(list.Items) == 0 {
			t.Fatal("expected completions, got none")
		}
		env.OpenFile("a/b.go")
		if list := env.Completion("a/b.go", env.RegexpSearch("a/b.go", "Println")); list == nil || len(list.Items) == 0 {
			t.Fatal("expected completions, got none")
		}
		if list := env.Completion("a/a.go", env.RegexpSearch("a/a.go", "Println")); list == nil || len(list.Items) == 0 {
			t.Fatal("expected completions, got none")
		}
	})
}
