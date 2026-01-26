// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"time"
)

func init() {
	register("PrintGOMAXPROCS", PrintGOMAXPROCS)
	register("SetLimitThenDefaultGOMAXPROCS", SetLimitThenDefaultGOMAXPROCS)
	register("UpdateGOMAXPROCS", UpdateGOMAXPROCS)
	register("DontUpdateGOMAXPROCS", DontUpdateGOMAXPROCS)
}

func PrintGOMAXPROCS() {
	println(runtime.GOMAXPROCS(0))
}

func mustSetCPUMax(path string, quota int64) {
	q := "max"
	if quota >= 0 {
		q = strconv.FormatInt(quota, 10)
	}
	buf := fmt.Sprintf("%s 100000", q)
	if err := os.WriteFile(path, []byte(buf), 0); err != nil {
		panic(fmt.Sprintf("error setting cpu.max: %v", err))
	}
}

func mustParseInt64(s string) int64 {
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		panic(err)
	}
	return v
}

// Inputs:
// GO_TEST_CPU_MAX_PATH: Path to cgroup v2 cpu.max file.
// GO_TEST_CPU_MAX_QUOTA: CPU quota to set.
func SetLimitThenDefaultGOMAXPROCS() {
	path := os.Getenv("GO_TEST_CPU_MAX_PATH")
	quota := mustParseInt64(os.Getenv("GO_TEST_CPU_MAX_QUOTA"))

	mustSetCPUMax(path, quota)

	runtime.SetDefaultGOMAXPROCS()
	println(runtime.GOMAXPROCS(0))
}

// Wait for GOMAXPROCS to change from from to to. Times out after 10s.
func waitForMaxProcsChange(from, to int) {
	start := time.Now()
	for {
		if time.Since(start) > 10*time.Second {
			panic("no update for >10s")
		}

		procs := runtime.GOMAXPROCS(0)
		println("GOMAXPROCS:", procs)
		if procs == to {
			return
		}
		if procs != from {
			panic(fmt.Sprintf("GOMAXPROCS change got %d want %d", procs, to))
		}

		time.Sleep(100*time.Millisecond)
	}
}

// Make sure that GOMAXPROCS does not change from curr.
//
// It is impossible to assert that it never changes, so this just makes sure it
// stays for 5s.
func mustNotChangeMaxProcs(curr int) {
	start := time.Now()
	for {
		if time.Since(start) > 5*time.Second {
			return
		}

		procs := runtime.GOMAXPROCS(0)
		println("GOMAXPROCS:", procs)
		if procs != curr {
			panic(fmt.Sprintf("GOMAXPROCS change got %d want %d", procs, curr))
		}

		time.Sleep(100*time.Millisecond)
	}
}

// Inputs:
// GO_TEST_CPU_MAX_PATH: Path to cgroup v2 cpu.max file.
func UpdateGOMAXPROCS() {
	// We start with no limit.

	ncpu := runtime.NumCPU()

	procs := runtime.GOMAXPROCS(0)
	println("GOMAXPROCS:", procs)
	if procs != ncpu {
		panic(fmt.Sprintf("GOMAXPROCS got %d want %d", procs, ncpu))
	}

	path := os.Getenv("GO_TEST_CPU_MAX_PATH")

	// Drop down to 3 CPU.
	mustSetCPUMax(path, 300000)
	waitForMaxProcsChange(ncpu, 3)

	// Drop even further. Now we hit the minimum GOMAXPROCS=2.
	mustSetCPUMax(path, 100000)
	waitForMaxProcsChange(3, 2)

	// Increase back up.
	mustSetCPUMax(path, 300000)
	waitForMaxProcsChange(2, 3)

	// Remove limit entirely.
	mustSetCPUMax(path, -1)
	waitForMaxProcsChange(3, ncpu)

	// Setting GOMAXPROCS explicitly disables updates.
	runtime.GOMAXPROCS(3)
	mustSetCPUMax(path, 200000)
	mustNotChangeMaxProcs(3)

	// Re-enable updates. Change is immediately visible.
	runtime.SetDefaultGOMAXPROCS()
	procs = runtime.GOMAXPROCS(0)
	println("GOMAXPROCS:", procs)
	if procs != 2 {
		panic(fmt.Sprintf("GOMAXPROCS got %d want %d", procs, 2))
	}

	// Setting GOMAXPROCS to itself also disables updates, despite not
	// changing the value itself.
	runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	mustSetCPUMax(path, 300000)
	mustNotChangeMaxProcs(2)

	println("OK")
}

// Inputs:
// GO_TEST_CPU_MAX_PATH: Path to cgroup v2 cpu.max file.
func DontUpdateGOMAXPROCS() {
	// The caller has disabled updates. Make sure they don't happen.

	curr := runtime.GOMAXPROCS(0)
	println("GOMAXPROCS:", curr)

	path := os.Getenv("GO_TEST_CPU_MAX_PATH")
	mustSetCPUMax(path, 300000)
	mustNotChangeMaxProcs(curr)

	println("OK")
}
