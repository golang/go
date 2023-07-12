// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diagnostics

import (
	"fmt"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// Test for the timeformat analyzer, following golang/vscode-go#2406.
//
// This test checks that applying the suggested fix from the analyzer resolves
// the diagnostic warning.
func TestTimeFormatAnalyzer(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main

import (
	"fmt"
	"time"
)

func main() {
	now := time.Now()
	fmt.Println(now.Format("2006-02-01"))
}`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")

		var d protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "2006-02-01")),
			ReadDiagnostics("main.go", &d),
		)

		env.ApplyQuickFixes("main.go", d.Diagnostics)
		env.AfterChange(NoDiagnostics(ForFile("main.go")))
	})
}

func TestAnalysisProgressReporting(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.18

-- main.go --
package main

func main() {
}`

	tests := []struct {
		setting bool
		want    Expectation
	}{
		{true, CompletedWork(cache.AnalysisProgressTitle, 1, true)},
		{false, Not(CompletedWork(cache.AnalysisProgressTitle, 1, true))},
	}

	for _, test := range tests {
		t.Run(fmt.Sprint(test.setting), func(t *testing.T) {
			WithOptions(
				Settings{
					"reportAnalysisProgressAfter": "0s",
					"analysisProgressReporting":   test.setting,
				},
			).Run(t, files, func(t *testing.T, env *Env) {
				env.OpenFile("main.go")
				env.AfterChange(test.want)
			})
		})
	}
}

// Test the embed directive analyzer.
//
// There is a fix for missing imports, but it should not trigger for other
// kinds of issues reported by the analayzer, here the variable
// declaration following the embed directive is wrong.
func TestNoSuggestedFixesForEmbedDirectiveDeclaration(t *testing.T) {
	const generated = `
-- go.mod --
module mod.com

go 1.20

-- foo.txt --
FOO

-- main.go --
package main

import _ "embed"

//go:embed foo.txt
var foo, bar string

func main() {
	_ = foo
}
`
	Run(t, generated, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.AfterChange(
			Diagnostics(env.AtRegexp("main.go", "//go:embed")),
			ReadDiagnostics("main.go", &d),
		)
		if fixes := env.GetQuickFixes("main.go", d.Diagnostics); len(fixes) != 0 {
			t.Errorf("got quick fixes %v, wanted none", fixes)
		}
	})
}
