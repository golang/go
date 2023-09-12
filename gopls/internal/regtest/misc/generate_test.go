// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rfindley): figure out why go generate fails on android builders.

//go:build !android
// +build !android

package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestGenerateProgress(t *testing.T) {
	const generatedWorkspace = `
-- go.mod --
module fake.test

go 1.14
-- generate.go --
// +build ignore

package main

import (
	"os"
)

func main() {
	os.WriteFile("generated.go", []byte("package " + os.Args[1] + "\n\nconst Answer = 21"), 0644)
}

-- lib1/lib.go --
package lib1

//` + `go:generate go run ../generate.go lib1

-- lib2/lib.go --
package lib2

//` + `go:generate go run ../generate.go lib2

-- main.go --
package main

import (
	"fake.test/lib1"
	"fake.test/lib2"
)

func main() {
	println(lib1.Answer + lib2.Answer)
}
`

	Run(t, generatedWorkspace, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("main.go", "lib1.(Answer)")),
		)
		env.RunGenerate("./lib1")
		env.RunGenerate("./lib2")
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
		)
	})
}
