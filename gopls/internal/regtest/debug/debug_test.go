// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestMain(m *testing.M) {
	Main(m, hooks.Options)
}

func TestBugNotification(t *testing.T) {
	// Verify that a properly configured session gets notified of a bug on the
	// server.
	WithOptions(
		Modes(Default), // must be in-process to receive the bug report below
		Settings{"showBugReports": true},
	).Run(t, "", func(t *testing.T, env *Env) {
		const desc = "got a bug"
		bug.Report(desc)
		env.Await(ShownMessage(desc))
	})
}

// TestStartDebugging executes a gopls.start_debugging command to
// start the internal web server.
func TestStartDebugging(t *testing.T) {
	WithOptions(
		Modes(Default|Experimental), // doesn't work in Forwarded mode
	).Run(t, "", func(t *testing.T, env *Env) {
		// Start a debugging server.
		res, err := startDebugging(env.Ctx, env.Editor.Server, &command.DebuggingArgs{
			Addr: "", // any free port
		})
		if err != nil {
			t.Fatalf("startDebugging: %v", err)
		}

		// Assert that the server requested that the
		// client show the debug page in a browser.
		debugURL := res.URLs[0]
		env.Await(ShownDocument(debugURL))

		// Send a request to the debug server and ensure it responds.
		resp, err := http.Get(debugURL)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("reading HTTP response body: %v", err)
		}
		const want = "<title>GoPls"
		if !strings.Contains(string(data), want) {
			t.Errorf("GET %s response does not contain %q: <<%s>>", debugURL, want, data)
		}
	})
}

// startDebugging starts a debugging server.
// TODO(adonovan): move into command package?
func startDebugging(ctx context.Context, server protocol.Server, args *command.DebuggingArgs) (*command.DebuggingResult, error) {
	rawArgs, err := command.MarshalArgs(args)
	if err != nil {
		return nil, err
	}
	res0, err := server.ExecuteCommand(ctx, &protocol.ExecuteCommandParams{
		Command:   command.StartDebugging.ID(),
		Arguments: rawArgs,
	})
	if err != nil {
		return nil, err
	}
	// res0 is the result of a schemaless (map[string]any) JSON decoding.
	// Re-encode and decode into the correct Go struct type.
	// TODO(adonovan): fix (*serverDispatcher).ExecuteCommand.
	data, err := json.Marshal(res0)
	if err != nil {
		return nil, err
	}
	var res *command.DebuggingResult
	if err := json.Unmarshal(data, &res); err != nil {
		return nil, err
	}
	return res, nil
}
