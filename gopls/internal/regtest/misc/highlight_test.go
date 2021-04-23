// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"sort"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestWorkspacePackageHighlight(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
	var A string = "A"
	x := "x-" + A
	println(A, x)
}`

	Run(t, mod, func(t *testing.T, env *Env) {
		const file = "main.go"
		env.OpenFile(file)
		_, pos := env.GoToDefinition(file, env.RegexpSearch(file, `var (A) string`))

		checkHighlights(env, file, pos, 3)
	})
}

func TestStdPackageHighlight_Issue43511(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Printf()
}`

	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		file, _ := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `fmt\.(Printf)`))
		pos := env.RegexpSearch(file, `func Printf\((format) string`)

		checkHighlights(env, file, pos, 2)
	})
}

func TestThirdPartyPackageHighlight_Issue43511(t *testing.T) {
	const proxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/global/global.go --
package global

const A = 1

func foo() {
	_ = A
}

func bar() int {
	return A + A
}
-- example.com@v1.2.3/local/local.go --
package local

func foo() int {
	const b = 2

	return b * b * (b+1) + b
}`

	const mod = `
-- go.mod --
module mod.com

go 1.12

require example.com v1.2.3
-- go.sum --
example.com v1.2.3 h1:WFzrgiQJwEDJNLDUOV1f9qlasQkvzXf2UNLaNIqbWsI=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- main.go --
package main

import (
	_ "example.com/global"
	_ "example.com/local"
)

func main() {}`

	WithOptions(
		ProxyFiles(proxy),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")

		file, _ := env.GoToDefinition("main.go", env.RegexpSearch("main.go", `"example.com/global"`))
		pos := env.RegexpSearch(file, `const (A)`)
		checkHighlights(env, file, pos, 4)

		file, _ = env.GoToDefinition("main.go", env.RegexpSearch("main.go", `"example.com/local"`))
		pos = env.RegexpSearch(file, `const (b)`)
		checkHighlights(env, file, pos, 5)
	})
}

func checkHighlights(env *Env, file string, pos fake.Pos, highlightCount int) {
	t := env.T
	t.Helper()

	highlights := env.DocumentHighlight(file, pos)
	if len(highlights) != highlightCount {
		t.Fatalf("expected %v highlight(s), got %v", highlightCount, len(highlights))
	}

	references := env.References(file, pos)
	if len(highlights) != len(references) {
		t.Fatalf("number of highlights and references is expected to be equal: %v != %v", len(highlights), len(references))
	}

	sort.Slice(highlights, func(i, j int) bool {
		return protocol.CompareRange(highlights[i].Range, highlights[j].Range) < 0
	})
	sort.Slice(references, func(i, j int) bool {
		return protocol.CompareRange(references[i].Range, references[j].Range) < 0
	})
	for i := range highlights {
		if highlights[i].Range != references[i].Range {
			t.Errorf("highlight and reference ranges are expected to be equal: %v != %v", highlights[i].Range, references[i].Range)
		}
	}
}
