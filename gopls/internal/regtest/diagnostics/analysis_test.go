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
