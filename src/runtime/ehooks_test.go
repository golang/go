// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"os/exec"
	"strings"
	"testing"
)

func TestExitHooks(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping due to -short")
	}

	scenarios := []struct {
		mode     string
		expected string
		musthave []string
	}{
		{
			mode:     "simple",
			expected: "bar foo",
		},
		{
			mode:     "goodexit",
			expected: "orange apple",
		},
		{
			mode:     "badexit",
			expected: "blub blix",
		},
		{
			mode: "panics",
			musthave: []string{
				"fatal error: exit hook invoked panic",
				"main.testPanics",
			},
		},
		{
			mode: "callsexit",
			musthave: []string{
				"fatal error: exit hook invoked exit",
			},
		},
		{
			mode:     "exit2",
			expected: "",
		},
	}

	exe, err := buildTestProg(t, "testexithooks")
	if err != nil {
		t.Fatal(err)
	}

	for _, s := range scenarios {
		cmd := exec.Command(exe, []string{"-mode", s.mode}...)
		out, _ := cmd.CombinedOutput()
		outs := strings.ReplaceAll(string(out), "\n", " ")
		outs = strings.TrimSpace(outs)
		if s.expected != "" && s.expected != outs {
			t.Fatalf("failed %s: wanted %q\noutput:\n%s",
			s.mode, s.expected, outs)
		}
		for _, need := range s.musthave {
			if !strings.Contains(outs, need) {
				t.Fatalf("failed mode %s: output does not contain %q\noutput:\n%s",
				s.mode, need, outs)
			}
		}
		if s.expected == "" && s.musthave == nil && outs != "" {
			t.Errorf("failed mode %s: wanted no output\noutput:\n%s", s.mode, outs)
		}
	}
}
