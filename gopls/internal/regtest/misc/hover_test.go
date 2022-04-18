// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	. "golang.org/x/tools/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

func TestHoverUnexported(t *testing.T) {
	const proxy = `
-- golang.org/x/structs@v1.0.0/go.mod --
module golang.org/x/structs

go 1.12

-- golang.org/x/structs@v1.0.0/types.go --
package structs

type Mixed struct {
	// Exported comment
	Exported   int
	unexported string
}

func printMixed(m Mixed) {
	println(m)
}
`
	const mod = `
-- go.mod --
module mod.com

go 1.12

require golang.org/x/structs v1.0.0
-- go.sum --
golang.org/x/structs v1.0.0 h1:Ito/a7hBYZaNKShFrZKjfBA/SIPvmBrcPCBWPx5QeKk=
golang.org/x/structs v1.0.0/go.mod h1:47gkSIdo5AaQaWJS0upVORsxfEr1LL1MWv9dmYF3iq4=
-- main.go --
package main

import "golang.org/x/structs"

func main() {
	var m structs.Mixed
	_ = m.Exported
}
`

	// TODO: use a nested workspace folder here.
	WithOptions(
		ProxyFiles(proxy),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		mixedPos := env.RegexpSearch("main.go", "Mixed")
		got, _ := env.Hover("main.go", mixedPos)
		if !strings.Contains(got.Value, "unexported") {
			t.Errorf("Workspace hover: missing expected field 'unexported'. Got:\n%q", got.Value)
		}

		cacheFile, _ := env.GoToDefinition("main.go", mixedPos)
		argPos := env.RegexpSearch(cacheFile, "printMixed.*(Mixed)")
		got, _ = env.Hover(cacheFile, argPos)
		if !strings.Contains(got.Value, "unexported") {
			t.Errorf("Non-workspace hover: missing expected field 'unexported'. Got:\n%q", got.Value)
		}

		exportedFieldPos := env.RegexpSearch("main.go", "Exported")
		got, _ = env.Hover("main.go", exportedFieldPos)
		if !strings.Contains(got.Value, "comment") {
			t.Errorf("Workspace hover: missing comment for field 'Exported'. Got:\n%q", got.Value)
		}
	})
}

func TestHoverIntLiteral(t *testing.T) {
	testenv.NeedsGo1Point(t, 13)
	const source = `
-- main.go --
package main

var (
	bigBin = 0b1001001
)

var hex = 0xe34e

func main() {
}
`
	Run(t, source, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		hexExpected := "58190"
		got, _ := env.Hover("main.go", env.RegexpSearch("main.go", "hex"))
		if got != nil && !strings.Contains(got.Value, hexExpected) {
			t.Errorf("Hover: missing expected field '%s'. Got:\n%q", hexExpected, got.Value)
		}

		binExpected := "73"
		got, _ = env.Hover("main.go", env.RegexpSearch("main.go", "bigBin"))
		if got != nil && !strings.Contains(got.Value, binExpected) {
			t.Errorf("Hover: missing expected field '%s'. Got:\n%q", binExpected, got.Value)
		}
	})
}

// Tests that hovering does not trigger the panic in golang/go#48249.
func TestPanicInHoverBrokenCode(t *testing.T) {
	testenv.NeedsGo1Point(t, 13)
	const source = `
-- main.go --
package main

type Example struct`
	Run(t, source, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.Editor.Hover(env.Ctx, "main.go", env.RegexpSearch("main.go", "Example"))
	})
}

func TestHoverRune_48492(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.EditBuffer("main.go", fake.NewEdit(0, 0, 1, 0, "package main\nfunc main() {\nconst x = `\nfoo\n`\n}"))
		env.Editor.Hover(env.Ctx, "main.go", env.RegexpSearch("main.go", "foo"))
	})
}

func TestHoverImport(t *testing.T) {
	// For Go.13 and earlier versions, Go will try to download imported but missing packages. This behavior breaks the
	// workspace as Go fails to download non-existent package "mod.com/lib4"
	testenv.NeedsGo1Point(t, 14)
	const packageDoc1 = "Package lib1 hover documentation"
	const packageDoc2 = "Package lib2 hover documentation"
	tests := []struct {
		hoverPackage string
		want         string
	}{
		{
			"mod.com/lib1",
			packageDoc1,
		},
		{
			"mod.com/lib2",
			packageDoc2,
		},
		{
			"mod.com/lib3",
			"",
		},
	}
	source := fmt.Sprintf(`
-- go.mod --
module mod.com

go 1.12
-- lib1/a.go --
// %s
package lib1

const C = 1

-- lib1/b.go --
package lib1

const D = 1

-- lib2/a.go --
// %s
package lib2

const E = 1

-- lib3/a.go --
package lib3

const F = 1

-- main.go --
package main

import (
	"mod.com/lib1"
	"mod.com/lib2"
	"mod.com/lib3"
	"mod.com/lib4"
)

func main() {
	println("Hello")
}
	`, packageDoc1, packageDoc2)
	Run(t, source, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		for _, test := range tests {
			got, _ := env.Hover("main.go", env.RegexpSearch("main.go", test.hoverPackage))
			if !strings.Contains(got.Value, test.want) {
				t.Errorf("Hover: got:\n%q\nwant:\n%q", got.Value, test.want)
			}
		}

		got, _ := env.Hover("main.go", env.RegexpSearch("main.go", "mod.com/lib4"))
		if got != nil {
			t.Errorf("Hover: got:\n%q\nwant:\n%v", got.Value, nil)
		}
	})
}
