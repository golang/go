// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rfindley): figure out why go generate fails on android builders.

// +build !android

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp"
)

func TestGenerateProgress(t *testing.T) {
	const generatedWorkspace = `
-- go.mod --
module fake.test

go 1.14
-- generate.go --
// +build ignore

package main

import "io/ioutil"

func main() {
	ioutil.WriteFile("generated.go", []byte("package lib\n\nconst answer = 42"), 0644)
}
-- lib.go --
package lib

func GetAnswer() int {
	return answer
}

//go:generate go run generate.go
`

	runner.Run(t, generatedWorkspace, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("lib.go", "answer"),
		)
		env.RunGenerate(".")
		env.Await(
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
				EmptyDiagnostics("lib.go")),
		)
	})
}
