// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package runtime_test

import (
	"internal/race"
	"runtime"
	"testing"
)

func TestNotInGoMetric(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skip("unsupported on Windows and Plan9")
	case "freebsd":
		if race.Enabled {
			t.Skipf("race + cgo freebsd not supported. See https://go.dev/issue/73788.")
		}
	}

	run := func(t *testing.T, name string) {
		// This test is run in a subprocess to prevent other tests from polluting the metrics
		// and because we need to make some cgo callbacks.
		output := runTestProg(t, "testprogcgo", name)
		want := "OK\n"
		if output != want {
			t.Fatalf("output:\n%s\n\nwanted:\n%s", output, want)
		}
	}
	t.Run("CgoCall", func(t *testing.T) {
		run(t, "NotInGoMetricCgoCall")
	})
	t.Run("CgoCallback", func(t *testing.T) {
		run(t, "NotInGoMetricCgoCallback")
	})
	t.Run("CgoCallAndCallback", func(t *testing.T) {
		run(t, "NotInGoMetricCgoCallAndCallback")
	})
}
