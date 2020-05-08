// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"internal/testenv"
	"os/exec"
	"testing"
)

// BenchmarkExecGoEnv measures how long it takes for 'go env GOARCH' to run.
// Since 'go' is executed, remember to run 'go install cmd/go' before running
// the benchmark if any changes were done.
func BenchmarkExecGoEnv(b *testing.B) {
	testenv.MustHaveExec(b)
	b.StopTimer()
	gotool, err := testenv.GoTool()
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < b.N; i++ {
		cmd := exec.Command(gotool, "env", "GOARCH")

		b.StartTimer()
		err := cmd.Run()
		b.StopTimer()

		if err != nil {
			b.Fatal(err)
		}
	}
}
