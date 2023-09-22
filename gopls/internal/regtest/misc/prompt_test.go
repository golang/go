// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// Test that gopls prompts for telemetry only when it is supposed to.
func TestTelemetryPrompt_Conditions(t *testing.T) {
	const src = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
}
`

	for _, enabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("telemetryPrompt=%v", enabled), func(t *testing.T) {
			for _, initialMode := range []string{"", "off", "on"} {
				t.Run(fmt.Sprintf("initial_mode=%s", initialMode), func(t *testing.T) {
					modeFile := filepath.Join(t.TempDir(), "mode")
					if initialMode != "" {
						if err := os.WriteFile(modeFile, []byte(initialMode), 0666); err != nil {
							t.Fatal(err)
						}
					}
					WithOptions(
						Modes(Default), // no need to run this in all modes
						EnvVars{
							lsp.GoplsConfigDirEnvvar:        t.TempDir(),
							lsp.FakeTelemetryModefileEnvvar: modeFile,
						},
						Settings{
							"telemetryPrompt": enabled,
						},
					).Run(t, src, func(t *testing.T, env *Env) {
						wantPrompt := enabled && (initialMode == "" || initialMode == "off")
						expectation := ShownMessageRequest(".*Would you like to enable Go telemetry?")
						if !wantPrompt {
							expectation = Not(expectation)
						}
						env.OnceMet(
							CompletedWork(lsp.TelemetryPromptWorkTitle, 1, true),
							expectation,
						)
					})
				})
			}
		})
	}
}

// Test that responding to the telemetry prompt results in the expected state.
func TestTelemetryPrompt_Response(t *testing.T) {
	const src = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
}
`

	tests := []struct {
		response string // response to choose for the telemetry dialog
		wantMode string // resulting telemetry mode
		wantMsg  string // substring contained in the follow-up popup (if empty, no popup is expected)
	}{
		{lsp.TelemetryYes, "on", "uploading is now enabled"},
		{lsp.TelemetryNo, "", ""},
		{"", "", ""},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("response=%s", test.response), func(t *testing.T) {
			modeFile := filepath.Join(t.TempDir(), "mode")
			msgRE := regexp.MustCompile(".*Would you like to enable Go telemetry?")
			respond := func(m *protocol.ShowMessageRequestParams) (*protocol.MessageActionItem, error) {
				if msgRE.MatchString(m.Message) {
					for _, item := range m.Actions {
						if item.Title == test.response {
							return &item, nil
						}
					}
					if test.response != "" {
						t.Errorf("action item %q not found", test.response)
					}
				}
				return nil, nil
			}
			WithOptions(
				Modes(Default), // no need to run this in all modes
				EnvVars{
					lsp.GoplsConfigDirEnvvar:        t.TempDir(),
					lsp.FakeTelemetryModefileEnvvar: modeFile,
				},
				Settings{
					"telemetryPrompt": true,
				},
				MessageResponder(respond),
			).Run(t, src, func(t *testing.T, env *Env) {
				var postConditions []Expectation
				if test.wantMsg != "" {
					postConditions = append(postConditions, ShownMessage(test.wantMsg))
				}
				env.OnceMet(
					CompletedWork(lsp.TelemetryPromptWorkTitle, 1, true),
					postConditions...,
				)
				gotMode := ""
				if contents, err := os.ReadFile(modeFile); err == nil {
					gotMode = string(contents)
				} else if !os.IsNotExist(err) {
					t.Fatal(err)
				}
				if gotMode != test.wantMode {
					t.Errorf("after prompt, mode=%s, want %s", gotMode, test.wantMode)
				}
			})
		})
	}
}

// Test that we stop asking about telemetry after the user ignores the question
// 5 times.
func TestTelemetryPrompt_GivingUp(t *testing.T) {
	const src = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
}
`

	// For this test, we want to share state across gopls sessions.
	modeFile := filepath.Join(t.TempDir(), "mode")
	configDir := t.TempDir()

	const maxPrompts = 5 // internal prompt limit defined by gopls

	for i := 0; i < maxPrompts+1; i++ {
		WithOptions(
			Modes(Default), // no need to run this in all modes
			EnvVars{
				lsp.GoplsConfigDirEnvvar:        configDir,
				lsp.FakeTelemetryModefileEnvvar: modeFile,
			},
			Settings{
				"telemetryPrompt": true,
			},
		).Run(t, src, func(t *testing.T, env *Env) {
			wantPrompt := i < maxPrompts
			expectation := ShownMessageRequest(".*Would you like to enable Go telemetry?")
			if !wantPrompt {
				expectation = Not(expectation)
			}
			env.OnceMet(
				CompletedWork(lsp.TelemetryPromptWorkTitle, 1, true),
				expectation,
			)
		})
	}
}

// Test that gopls prompts for telemetry only when it is supposed to.
func TestTelemetryPrompt_Conditions2(t *testing.T) {
	const src = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
}
`
	modeFile := filepath.Join(t.TempDir(), "mode")
	WithOptions(
		Modes(Default), // no need to run this in all modes
		EnvVars{
			lsp.GoplsConfigDirEnvvar:        t.TempDir(),
			lsp.FakeTelemetryModefileEnvvar: modeFile,
		},
		Settings{
			// off because we are testing
			// if we can trigger the prompt with command.
			"telemetryPrompt": false,
		},
	).Run(t, src, func(t *testing.T, env *Env) {
		cmd, err := command.NewMaybePromptForTelemetryCommand("prompt")
		if err != nil {
			t.Fatal(err)
		}
		var result error
		env.ExecuteCommand(&protocol.ExecuteCommandParams{
			Command: cmd.Command,
		}, &result)
		if result != nil {
			t.Fatal(err)
		}
		expectation := ShownMessageRequest(".*Would you like to enable Go telemetry?")
		env.OnceMet(
			CompletedWork(lsp.TelemetryPromptWorkTitle, 2, true),
			expectation,
		)
	})
}
