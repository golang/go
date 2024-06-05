// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/platform"
	"internal/testenv"
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

func TestExitHooks(t *testing.T) {
	bmodes := []string{""}
	if testing.Short() {
		t.Skip("skipping due to -short")
	}
	// Note the HasCGO() test below; this is to prevent the test
	// running if CGO_ENABLED=0 is in effect.
	haverace := platform.RaceDetectorSupported(runtime.GOOS, runtime.GOARCH)
	if haverace && testenv.HasCGO() {
		bmodes = append(bmodes, "-race")
	}
	for _, bmode := range bmodes {
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

		exe, err := buildTestProg(t, "testexithooks", bmode)
		if err != nil {
			t.Fatal(err)
		}

		bt := ""
		if bmode != "" {
			bt = " bmode: " + bmode
		}
		for _, s := range scenarios {
			cmd := exec.Command(exe, []string{"-mode", s.mode}...)
			out, _ := cmd.CombinedOutput()
			outs := strings.ReplaceAll(string(out), "\n", " ")
			outs = strings.TrimSpace(outs)
			if s.expected != "" && s.expected != outs {
				t.Fatalf("failed%s mode %s: wanted %q\noutput:\n%s", bt,
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
}
