// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"internal/testenv"
	"path/filepath"
	"sync/atomic"
	"testing"
)

var modulesForTest = []struct {
	name           string
	testDataFolder string
}{
	{
		name:           "Empty",
		testDataFolder: "empty",
	},
	{
		name:           "Cmd",
		testDataFolder: "cmd",
	},
	{
		name:           "K8S",
		testDataFolder: "strippedk8s",
	},
}

func BenchmarkListModules(b *testing.B) {
	testenv.MustHaveExec(b)
	gotool, err := testenv.GoTool()
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for _, m := range modulesForTest {
		// Do not run in parallel. GOPROXY rate limits may affect parallel executions.
		b.Run(m.name, func(b *testing.B) {
			// We collect extra metrics.
			var n, userTime, systemTime int64
			modDir := filepath.Join("testdata", "list", m.testDataFolder, "go.mod")
			for i := 0; i < b.N; i++ {
				cmd := testenv.Command(b, gotool, "list", "-m", "-modfile="+modDir, "-mod=readonly", "all")

				// Guarantees clean module cache for every execution.
				gopath := b.TempDir()
				cmd.Env = append(cmd.Env, "GOPATH="+gopath)

				if err := cmd.Run(); err != nil {
					b.Fatal(err)
				}
				atomic.AddInt64(&n, 1)
				atomic.AddInt64(&userTime, int64(cmd.ProcessState.UserTime()))
				atomic.AddInt64(&systemTime, int64(cmd.ProcessState.SystemTime()))

			}
			b.ReportMetric(float64(userTime)/float64(n), "user-ns/op")
			b.ReportMetric(float64(systemTime)/float64(n), "sys-ns/op")
		})
	}
}
