// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/cgrouptest"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

func mustHaveFourCPUs(t *testing.T) {
	// If NumCPU is lower than the cgroup limit, GOMAXPROCS will use
	// NumCPU.
	//
	// cgroup GOMAXPROCS also have a minimum of 2. We need some room above
	// that to test interesting properies.
	if runtime.NumCPU() < 4 {
		t.Helper()
		t.Skip("skipping test: fewer than 4 CPUs")
	}
}

func TestCgroupGOMAXPROCS(t *testing.T) {
	mustHaveFourCPUs(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		godebug int
		want    int
	}{
		// With containermaxprocs=1, GOMAXPROCS should use the cgroup
		// limit.
		{
			godebug: 1,
			want:    3,
		},
		// With containermaxprocs=0, it should be ignored.
		{
			godebug: 0,
			want:    runtime.NumCPU(),
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("containermaxprocs=%d", tc.godebug), func(t *testing.T) {
			cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
				if err := c.SetCPUMax(300000, 100000); err != nil {
					t.Fatalf("unable to set CPU limit: %v", err)
				}

				got := runBuiltTestProg(t, exe, "PrintGOMAXPROCS", fmt.Sprintf("GODEBUG=containermaxprocs=%d", tc.godebug))
				want := fmt.Sprintf("%d\n", tc.want)
				if got != want {
					t.Fatalf("output got %q want %q", got, want)
				}
			})
		})
	}
}

// Without a cgroup limit, GOMAXPROCS uses NumCPU.
func TestCgroupGOMAXPROCSNoLimit(t *testing.T) {
	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
		if err := c.SetCPUMax(-1, 100000); err != nil {
			t.Fatalf("unable to set CPU limit: %v", err)
		}

		got := runBuiltTestProg(t, exe, "PrintGOMAXPROCS")
		want := fmt.Sprintf("%d\n", runtime.NumCPU())
		if got != want {
			t.Fatalf("output got %q want %q", got, want)
		}
	})
}

// If the cgroup limit is higher than NumCPU, GOMAXPROCS uses NumCPU.
func TestCgroupGOMAXPROCSHigherThanNumCPU(t *testing.T) {
	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
		if err := c.SetCPUMax(2*int64(runtime.NumCPU())*100000, 100000); err != nil {
			t.Fatalf("unable to set CPU limit: %v", err)
		}

		got := runBuiltTestProg(t, exe, "PrintGOMAXPROCS")
		want := fmt.Sprintf("%d\n", runtime.NumCPU())
		if got != want {
			t.Fatalf("output got %q want %q", got, want)
		}
	})
}

func TestCgroupGOMAXPROCSRound(t *testing.T) {
	mustHaveFourCPUs(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		quota int64
		want  int
	}{
		// We always round the fractional component up.
		{
			quota: 200001,
			want:  3,
		},
		{
			quota: 250000,
			want:  3,
		},
		{
			quota: 299999,
			want:  3,
		},
		// Anything less than two rounds up to a minimum of 2.
		{
			quota: 50000, // 0.5
			want:  2,
		},
		{
			quota: 100000,
			want:  2,
		},
		{
			quota: 150000,
			want:  2,
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("%d", tc.quota), func(t *testing.T) {
			cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
				if err := c.SetCPUMax(tc.quota, 100000); err != nil {
					t.Fatalf("unable to set CPU limit: %v", err)
				}

				got := runBuiltTestProg(t, exe, "PrintGOMAXPROCS")
				want := fmt.Sprintf("%d\n", tc.want)
				if got != want {
					t.Fatalf("output got %q want %q", got, want)
				}
			})
		})
	}
}

// Environment variable takes precedence over defaults.
func TestCgroupGOMAXPROCSEnvironment(t *testing.T) {
	mustHaveFourCPUs(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
		if err := c.SetCPUMax(200000, 100000); err != nil {
			t.Fatalf("unable to set CPU limit: %v", err)
		}

		got := runBuiltTestProg(t, exe, "PrintGOMAXPROCS", "GOMAXPROCS=3")
		want := "3\n"
		if got != want {
			t.Fatalf("output got %q want %q", got, want)
		}
	})
}

// CPU affinity takes priority if lower than cgroup limit.
func TestCgroupGOMAXPROCSSchedAffinity(t *testing.T) {
	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
		if err := c.SetCPUMax(300000, 100000); err != nil {
			t.Fatalf("unable to set CPU limit: %v", err)
		}

		// CPU affinity is actually a per-thread attribute.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		const maxCPUs = 64 * 1024
		var orig [maxCPUs / 8]byte
		_, _, errno := syscall.Syscall6(syscall.SYS_SCHED_GETAFFINITY, 0, unsafe.Sizeof(orig), uintptr(unsafe.Pointer(&orig[0])), 0, 0, 0)
		if errno != 0 {
			t.Fatalf("unable to get CPU affinity: %v", errno)
		}

		// We're going to restrict to CPUs 0 and 1. Make sure those are already available.
		if orig[0]&0b11 != 0b11 {
			t.Skipf("skipping test: CPUs 0 and 1 not available")
		}

		var mask [maxCPUs / 8]byte
		mask[0] = 0b11
		_, _, errno = syscall.Syscall6(syscall.SYS_SCHED_SETAFFINITY, 0, unsafe.Sizeof(mask), uintptr(unsafe.Pointer(&mask[0])), 0, 0, 0)
		if errno != 0 {
			t.Fatalf("unable to set CPU affinity: %v", errno)
		}
		defer func() {
			_, _, errno = syscall.Syscall6(syscall.SYS_SCHED_SETAFFINITY, 0, unsafe.Sizeof(orig), uintptr(unsafe.Pointer(&orig[0])), 0, 0, 0)
			if errno != 0 {
				t.Fatalf("unable to restore CPU affinity: %v", errno)
			}
		}()

		got := runBuiltTestProg(t, exe, "PrintGOMAXPROCS")
		want := "2\n"
		if got != want {
			t.Fatalf("output got %q want %q", got, want)
		}
	})
}

func TestCgroupGOMAXPROCSSetDefault(t *testing.T) {
	mustHaveFourCPUs(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		godebug int
		want    int
	}{
		// With containermaxprocs=1, SetDefaultGOMAXPROCS should observe
		// the cgroup limit.
		{
			godebug: 1,
			want:    3,
		},
		// With containermaxprocs=0, it should be ignored.
		{
			godebug: 0,
			want:    runtime.NumCPU(),
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("containermaxprocs=%d", tc.godebug), func(t *testing.T) {
			cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
				env := []string{
					fmt.Sprintf("GO_TEST_CPU_MAX_PATH=%s", c.CPUMaxPath()),
					"GO_TEST_CPU_MAX_QUOTA=300000",
					fmt.Sprintf("GODEBUG=containermaxprocs=%d", tc.godebug),
				}
				got := runBuiltTestProg(t, exe, "SetLimitThenDefaultGOMAXPROCS", env...)
				want := fmt.Sprintf("%d\n", tc.want)
				if got != want {
					t.Fatalf("output got %q want %q", got, want)
				}
			})
		})
	}
}

func TestCgroupGOMAXPROCSUpdate(t *testing.T) {
	mustHaveFourCPUs(t)

	if testing.Short() {
		t.Skip("skipping test: long sleeps")
	}

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
		got := runBuiltTestProg(t, exe, "UpdateGOMAXPROCS", fmt.Sprintf("GO_TEST_CPU_MAX_PATH=%s", c.CPUMaxPath()))
		if !strings.Contains(got, "OK") {
			t.Fatalf("output got %q want OK", got)
		}
	})
}

func TestCgroupGOMAXPROCSDontUpdate(t *testing.T) {
	mustHaveFourCPUs(t)

	if testing.Short() {
		t.Skip("skipping test: long sleeps")
	}

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	// Two ways to disable updates: explicit GOMAXPROCS or GODEBUG for
	// update feature.
	for _, v := range []string{"GOMAXPROCS=4", "GODEBUG=updatemaxprocs=0"} {
		t.Run(v, func(t *testing.T) {
			cgrouptest.InCgroupV2(t, func(c *cgrouptest.CgroupV2) {
				got := runBuiltTestProg(t, exe, "DontUpdateGOMAXPROCS",
					fmt.Sprintf("GO_TEST_CPU_MAX_PATH=%s", c.CPUMaxPath()),
					v)
				if !strings.Contains(got, "OK") {
					t.Fatalf("output got %q want OK", got)
				}
			})
		})
	}
}
