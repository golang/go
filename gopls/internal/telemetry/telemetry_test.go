// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21 && !openbsd && !js && !wasip1 && !solaris && !android && !386
// +build go1.21,!openbsd,!js,!wasip1,!solaris,!android,!386

package telemetry_test

import (
	"context"
	"errors"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/telemetry/counter"
	"golang.org/x/telemetry/counter/countertest" // requires go1.21+
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/gopls/internal/telemetry"
)

func TestMain(m *testing.M) {
	tmp, err := os.MkdirTemp("", "gopls-telemetry-test")
	if err != nil {
		panic(err)
	}
	countertest.Open(tmp)
	defer os.RemoveAll(tmp)
	Main(m, hooks.Options)
}

func TestTelemetry(t *testing.T) {
	var (
		goversion = ""
		editor    = "vscode" // We set ClientName("Visual Studio Code") below.
	)

	// Run gopls once to determine the Go version.
	WithOptions(
		Modes(Default),
	).Run(t, "", func(_ *testing.T, env *Env) {
		goversion = strconv.Itoa(env.GoVersion())
	})

	// counters that should be incremented once per session
	sessionCounters := []*counter.Counter{
		counter.New("gopls/client:" + editor),
		counter.New("gopls/goversion:1." + goversion),
		counter.New("fwd/vscode/linter:a"),
	}
	initialCounts := make([]uint64, len(sessionCounters))
	for i, c := range sessionCounters {
		count, err := countertest.ReadCounter(c)
		if err != nil {
			t.Fatalf("ReadCounter(%s): %v", c.Name(), err)
		}
		initialCounts[i] = count
	}

	// Verify that a properly configured session gets notified of a bug on the
	// server.
	WithOptions(
		Modes(Default), // must be in-process to receive the bug report below
		Settings{"showBugReports": true},
		ClientName("Visual Studio Code"),
	).Run(t, "", func(_ *testing.T, env *Env) {
		goversion = strconv.Itoa(env.GoVersion())
		addForwardedCounters(env, []string{"vscode/linter:a"}, []int64{1})
		const desc = "got a bug"
		bug.Report(desc) // want a stack counter with the trace starting from here.
		env.Await(ShownMessage(desc))
	})

	// gopls/editor:client
	// gopls/goversion:1.x
	// fwd/vscode/linter:a
	for i, c := range sessionCounters {
		want := initialCounts[i] + 1
		got, err := countertest.ReadCounter(c)
		if err != nil || got != want {
			t.Errorf("ReadCounter(%q) = (%v, %v), want (%v, nil)", c.Name(), got, err, want)
			t.Logf("Current timestamp = %v", time.Now().UTC())
		}
	}

	// gopls/bug
	bugcount := bug.BugReportCount
	counts, err := countertest.ReadStackCounter(bugcount)
	if err != nil {
		t.Fatalf("ReadStackCounter(bugreportcount) failed - %v", err)
	}
	if len(counts) != 1 || !hasEntry(counts, t.Name(), 1) {
		t.Errorf("read stackcounter(%q) = (%#v, %v), want one entry", "gopls/bug", counts, err)
		t.Logf("Current timestamp = %v", time.Now().UTC())
	}
}

func addForwardedCounters(env *Env, names []string, values []int64) {
	args, err := command.MarshalArgs(command.AddTelemetryCountersArgs{
		Names: names, Values: values,
	})
	if err != nil {
		env.T.Fatal(err)
	}
	var res error
	env.ExecuteCommand(&protocol.ExecuteCommandParams{
		Command:   command.AddTelemetryCounters.ID(),
		Arguments: args,
	}, res)
	if res != nil {
		env.T.Errorf("%v failed - %v", command.AddTelemetryCounters.ID(), res)
	}
}

func hasEntry(counts map[string]uint64, pattern string, want uint64) bool {
	for k, v := range counts {
		if strings.Contains(k, pattern) && v == want {
			return true
		}
	}
	return false
}

func TestLatencyCounter(t *testing.T) {
	const operation = "TestLatencyCounter" // a unique operation name

	stop := telemetry.StartLatencyTimer(operation)
	stop(context.Background(), nil)

	for isError, want := range map[bool]uint64{false: 1, true: 0} {
		if got := totalLatencySamples(t, operation, isError); got != want {
			t.Errorf("totalLatencySamples(operation=%v, isError=%v) = %d, want %d", operation, isError, got, want)
		}
	}
}

func TestLatencyCounter_Error(t *testing.T) {
	const operation = "TestLatencyCounter_Error" // a unique operation name

	stop := telemetry.StartLatencyTimer(operation)
	stop(context.Background(), errors.New("bad"))

	for isError, want := range map[bool]uint64{false: 0, true: 1} {
		if got := totalLatencySamples(t, operation, isError); got != want {
			t.Errorf("totalLatencySamples(operation=%v, isError=%v) = %d, want %d", operation, isError, got, want)
		}
	}
}

func TestLatencyCounter_Cancellation(t *testing.T) {
	const operation = "TestLatencyCounter_Cancellation"

	stop := telemetry.StartLatencyTimer(operation)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stop(ctx, nil)

	for isError, want := range map[bool]uint64{false: 0, true: 0} {
		if got := totalLatencySamples(t, operation, isError); got != want {
			t.Errorf("totalLatencySamples(operation=%v, isError=%v) = %d, want %d", operation, isError, got, want)
		}
	}
}

func totalLatencySamples(t *testing.T, operation string, isError bool) uint64 {
	var total uint64
	telemetry.ForEachLatencyCounter(operation, isError, func(c *counter.Counter) {
		count, err := countertest.ReadCounter(c)
		if err != nil {
			t.Errorf("ReadCounter(%s) failed: %v", c.Name(), err)
		} else {
			total += count
		}
	})
	return total
}

func TestLatencyInstrumentation(t *testing.T) {
	const files = `
-- go.mod --
module mod.test/a
go 1.18
-- a.go --
package a

func _() {
	x := 0
	_ = x
}
`

	// Verify that a properly configured session gets notified of a bug on the
	// server.
	WithOptions(
		Modes(Default), // must be in-process to receive the bug report below
	).Run(t, files, func(_ *testing.T, env *Env) {
		env.OpenFile("a.go")
		before := totalLatencySamples(t, "completion", false)
		loc := env.RegexpSearch("a.go", "x")
		for i := 0; i < 10; i++ {
			env.Completion(loc)
		}
		after := totalLatencySamples(t, "completion", false)
		if after-before < 10 {
			t.Errorf("after 10 completions, completion counter went from %d to %d", before, after)
		}
	})
}
