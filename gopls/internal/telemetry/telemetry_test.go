// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21 && !openbsd && !js && !wasip1 && !solaris && !android && !386
// +build go1.21,!openbsd,!js,!wasip1,!solaris,!android,!386

package telemetry_test

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/telemetry/counter"
	"golang.org/x/telemetry/counter/countertest" // requires go1.21+
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
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

	// Verify that a properly configured session gets notified of a bug on the
	// server.
	WithOptions(
		Modes(Default), // must be in-process to receive the bug report below
		Settings{"showBugReports": true},
		ClientName("Visual Studio Code"),
	).Run(t, "", func(t *testing.T, env *Env) {
		goversion = strconv.Itoa(env.GoVersion())
		const desc = "got a bug"
		bug.Report(desc) // want a stack counter with the trace starting from here.
		env.Await(ShownMessage(desc))
	})

	// gopls/editor:client
	// gopls/goversion:1.x
	for _, c := range []*counter.Counter{
		counter.New("gopls/client:" + editor),
		counter.New("gopls/goversion:1." + goversion),
	} {
		count, err := countertest.ReadCounter(c)
		if err != nil || count != 1 {
			t.Errorf("ReadCounter(%q) = (%v, %v), want (1, nil)", c.Name(), count, err)
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
