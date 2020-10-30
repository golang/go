// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"strings"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func TestHoverAndDocumentLink(t *testing.T) {
	testenv.NeedsGo1Point(t, 13)
	const program = `
-- go.mod --
module mod.test

go 1.12

require import.test v1.2.3
-- go.sum --
import.test v1.2.3 h1:Mu4N9BICLJFxwwn8YNg6T3frkFWW1O7evXvo0HiRjBc=
import.test v1.2.3/go.mod h1:KooCN1g237upRg7irU7F+3oADn5tVClU8YYW4I1xhMk=
-- main.go --
package main

import "import.test/pkg"

func main() {
	println(pkg.Hello)
}`

	const proxy = `
-- import.test@v1.2.3/go.mod --
module import.test

go 1.12
-- import.test@v1.2.3/pkg/const.go --
package pkg

const Hello = "Hello"
`
	runner.Run(t, program, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OpenFile("go.mod")

		modLink := "https://pkg.go.dev/mod/import.test@v1.2.3?utm_source=gopls"
		pkgLink := "https://pkg.go.dev/import.test@v1.2.3/pkg?utm_source=gopls"

		// First, check that we get the expected links via hover and documentLink.
		content, _ := env.Hover("main.go", env.RegexpSearch("main.go", "pkg.Hello"))
		if content == nil || !strings.Contains(content.Value, pkgLink) {
			t.Errorf("hover: got %v in main.go, want contains %q", content, pkgLink)
		}
		content, _ = env.Hover("go.mod", env.RegexpSearch("go.mod", "import.test"))
		if content == nil || !strings.Contains(content.Value, pkgLink) {
			t.Errorf("hover: got %v in go.mod, want contains %q", content, pkgLink)
		}
		links := env.DocumentLink("main.go")
		if len(links) != 1 || links[0].Target != pkgLink {
			t.Errorf("documentLink: got %v for main.go, want link to %q", links, pkgLink)
		}
		links = env.DocumentLink("go.mod")
		if len(links) != 1 || links[0].Target != modLink {
			t.Errorf("documentLink: got %v for go.mod, want link to %q", links, modLink)
		}

		// Then change the environment to make these links private.
		env.ChangeEnv(map[string]string{"GOPRIVATE": "import.test"})

		// Finally, verify that the links are gone.
		content, _ = env.Hover("main.go", env.RegexpSearch("main.go", "pkg.Hello"))
		if content == nil || strings.Contains(content.Value, pkgLink) {
			t.Errorf("hover: got %v in main.go, want non-empty hover without %q", content, pkgLink)
		}
		content, _ = env.Hover("go.mod", env.RegexpSearch("go.mod", "import.test"))
		if content == nil || strings.Contains(content.Value, modLink) {
			t.Errorf("hover: got %v in go.mod, want contains %q", content, modLink)
		}
		links = env.DocumentLink("main.go")
		if len(links) != 0 {
			t.Errorf("documentLink: got %d document links for main.go, want 0\nlinks: %v", len(links), links)
		}
		links = env.DocumentLink("go.mod")
		if len(links) != 0 {
			t.Errorf("documentLink: got %d document links for go.mod, want 0\nlinks: %v", len(links), links)
		}
	}, WithProxyFiles(proxy))
}
