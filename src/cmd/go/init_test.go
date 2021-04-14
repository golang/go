// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"internal/testenv"
	"os/exec"
	"sync/atomic"
	"testing"
)

// BenchmarkExecGoEnv measures how long it takes for 'go env GOARCH' to run.
// Since 'go' is executed, remember to run 'go install cmd/go' before running
// the benchmark if any changes were done.
func BenchmarkExecGoEnv(b *testing.B) {
	testenv.MustHaveExec(b)
	gotool, err := testenv.GoTool()
	if err != nil {
		b.Fatal(err)
	}

	// We collect extra metrics.
	var n, userTime, systemTime int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cmd := exec.Command(gotool, "env", "GOARCH")

			if err := cmd.Run(); err != nil {
				b.Fatal(err)
			}
			atomic.AddInt64(&n, 1)
			atomic.AddInt64(&userTime, int64(cmd.ProcessState.UserTime()))
			atomic.AddInt64(&systemTime, int64(cmd.ProcessState.SystemTime()))
		}
	})
	b.ReportMetric(float64(userTime)/float64(n), "user-ns/op")
	b.ReportMetric(float64(systemTime)/float64(n), "sys-ns/op")
}
