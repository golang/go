// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"cmd/go/internal/logos/config"
	"cmd/internal/script"
	"io"
	"runtime"
	"strings"
	"testing"
)

func TestLogosHostAMD64Condition(t *testing.T) {
	cond := scriptConditions(t)["host-goamd64"]
	if cond == nil {
		t.Fatal("host-goamd64 condition is missing")
	}
	host, _ := config.ArchLevels("native")
	for _, env := range [][]string{
		{"GOARCH=amd64", "GOAMD64=v1"},
		{"GOARCH=amd64", "GOAMD64=v4"},
		{"GOARCH=arm64", "GOAMD64=v4"},
	} {
		s, err := script.NewState(t.Context(), t.TempDir(), env)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.CloseAndWait(io.Discard); err != nil {
				t.Error(err)
			}
		})
		for _, level := range []string{"v1", "v2", "v3", "v4"} {
			got, err := cond.Eval(s, level)
			want := runtime.GOARCH == "amd64" && host >= level
			if err != nil || got != want {
				t.Errorf("host-goamd64:%s with %v = %v, %v; host %s/%s wants %v", level, env, got, err, runtime.GOARCH, host, want)
			}
		}
		for _, level := range []string{"", "v0", "v5", "v10", "native"} {
			got, err := cond.Eval(s, level)
			if got || err == nil || !strings.Contains(err.Error(), "unsupported GOAMD64 level") {
				t.Errorf("host-goamd64:%s = %v, %v; want a descriptive invalid-level error", level, got, err)
			}
		}
	}
}
