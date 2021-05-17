// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"strings"
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestHoverUnexported(t *testing.T) {
	const proxy = `
-- golang.org/x/structs@v1.0.0/go.mod --
module golang.org/x/structs

go 1.12

-- golang.org/x/structs@v1.0.0/types.go --
package structs

type Mixed struct {
	Exported   int
	unexported string
}
`
	const mod = `
-- go.mod --
module mod.com

go 1.12

require golang.org/x/structs v1.0.0
-- go.sum --
golang.org/x/structs v1.0.0 h1:oxD5q25qV458xBbXf5+QX+Johgg71KFtwuJzt145c9A=
golang.org/x/structs v1.0.0/go.mod h1:47gkSIdo5AaQaWJS0upVORsxfEr1LL1MWv9dmYF3iq4=
-- main.go --
package main

import "golang.org/x/structs"

func main() {
	var _ structs.Mixed
}
`
	// TODO: use a nested workspace folder here.
	WithOptions(
		ProxyFiles(proxy),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		got, _ := env.Hover("main.go", env.RegexpSearch("main.go", "Mixed"))
		if !strings.Contains(got.Value, "unexported") {
			t.Errorf("Hover: missing expected field 'unexported'. Got:\n%q", got.Value)
		}
	})
}
