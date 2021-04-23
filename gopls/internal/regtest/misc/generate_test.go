// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rfindley): figure out why go generate fails on android builders.

//go:build !android
// +build !android

package misc

import (
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestGenerateProgress(t *testing.T) {
	const generatedWorkspace = `
-- go.mod --
module fake.test

go 1.14
-- lib/generate.go --
// +build ignore

package main

import "io/ioutil"

func main() {
	ioutil.WriteFile("generated.go", []byte("package lib\n\nconst answer = 42"), 0644)
}
-- lib/lib.go --
package lib

func GetAnswer() int {
	return answer
}

//go:generate go run generate.go
`

	Run(t, generatedWorkspace, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("lib/lib.go", "answer"),
		)
		env.RunGenerate("./lib")
		env.Await(
			OnceMet(
				env.DoneWithChangeWatchedFiles(),
				EmptyDiagnostics("lib/lib.go")),
		)
	})
}
