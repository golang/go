// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package simd_test

import (
	"flag"
	"fmt"
	"internal/cpu"
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"simd"
	"slices"
	"strings"
	"testing"
)

var (
	listGODEBUGs = flag.Bool("list-godebugs", false, "print the GODEBUG settings to also run the tests under, then exit")
	runGODEBUGs  = flag.Bool("run-godebugs", false, "run the tests under the current GODEBUG setting and each one that -list-godebugs prints")
	printConfig  = flag.Bool("print-config", false, "print how the simd package is configured, then exit")
)

func TestMain(m *testing.M) {
	flag.Parse()
	switch {
	case *listGODEBUGs:
		// "go tool dist test" runs the tests again under each of
		// these. It looks for lines that start with "GODEBUG=".
		for _, s := range godebugSettings() {
			if !s.current {
				fmt.Println("GODEBUG=" + s.godebug)
			}
		}
	case *runGODEBUGs:
		os.Exit(runUnderGODEBUGs())
	case *printConfig:
		fmt.Println(config())
	default:
		os.Exit(m.Run())
	}
}

// runUnderGODEBUGs runs the test binary under each of the godebugSettings,
// with the same flags as this process, and returns the exit status for this
// process: 0 if all of the runs passed.
//
// This is "go test simd -run-godebugs", a way to test every implementation
// with one command. "go tool dist test" doesn't use it, because it wants a
// separate result for each test under each setting.
func runUnderGODEBUGs() int {
	exe, err := os.Executable()
	if err != nil {
		fmt.Println("FAIL: -run-godebugs:", err)
		return 1
	}
	// Strip the -run-godebugs flag.
	var args []string
	for _, arg := range os.Args[1:] {
		if name, _, _ := strings.Cut(strings.TrimLeft(arg, "-"), "="); name == "run-godebugs" {
			continue
		}
		args = append(args, arg)
	}
	status := 0
	for _, s := range godebugSettings() {
		fmt.Printf("# GODEBUG=%s\n", s.godebug)
		cmd := exec.Command(exe, args...)
		cmd.Env = append(os.Environ(), "GODEBUG="+s.godebug)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("FAIL: GODEBUG=%s: %v\n", s.godebug, err)
			status = 1
		}
	}
	return status
}

// A godebugSetting is a GODEBUG setting that selects one of the
// implementations of the simd package that this machine can run.
type godebugSetting struct {
	godebug string // the value of GODEBUG, including any settings it already has
	current bool   // this is the setting that this process is running under

	// wantConfig is the space-separated parts of config() that the
	// setting must produce. It is not set for the current setting.
	wantConfig string
}

// godebugSettings returns the GODEBUG settings that select each
// implementation of the simd package that this machine can run. The first is
// the current setting.
//
// The simd package selects an implementation based on the CPU, such as the
// largest vector size the CPU supports, so by default the tests only test
// that one. The others are what programs use on CPUs with fewer features.
// To test them too, "go tool dist test" uses the -list-godebugs flag to get
// the other settings and runs the tests again under each one. Which settings
// work depends on the CPU, which is why dist has to ask. The -run-godebugs
// flag runs the tests under all of them from a single "go test".
func godebugSettings() []godebugSetting {
	current := os.Getenv("GODEBUG")
	settings := []godebugSetting{{godebug: current, current: true}}
	if simd.Emulated() {
		// The emulation is the only implementation.
		return settings
	}
	// Each setting adds to the value that GODEBUG already has. Where
	// that value sets the same name, the addition wins, because it comes
	// later in the value.
	prefix := current
	if prefix != "" {
		prefix += ","
	}
	// Each smaller vector size, which CPUs with narrower vectors use.
	for bits := simd.VectorBits() / 2; bits >= 128; bits /= 2 {
		settings = append(settings, godebugSetting{
			godebug:    prefix + fmt.Sprintf("simd=%d", bits),
			wantConfig: fmt.Sprintf("VectorBits=%d Emulated=false", bits),
		})
	}
	// The emulation, which CPUs without SIMD use.
	settings = append(settings, godebugSetting{
		godebug:    prefix + "simd=0",
		wantConfig: "Emulated=true",
	})
	// The rest of the settings turn off CPU features.
	if !cpuOptionsWork() {
		return settings
	}
	// The fallbacks for carryless multiply, which archsimd uses on CPUs
	// that lack the instruction for it.
	switch runtime.GOARCH {
	case "amd64":
		// Each vector size has its own instruction and fallbacks. If
		// the CPU lacks an instruction, the tests already test its
		// fallback.
		for bits := simd.VectorBits(); bits >= 128; bits /= 2 {
			size := prefix + fmt.Sprintf("simd=%d,", bits)
			want := fmt.Sprintf("VectorBits=%d Emulated=false", bits)
			if bits > 128 && cpu.X86.HasVPCLMULQDQ {
				// Without VPCLMULQDQ, archsimd applies the Uint64x2
				// instruction to each 128-bit part of wider vectors.
				settings = append(settings, godebugSetting{godebug: size + "cpu.vpclmulqdq=off", wantConfig: want + " CLMUL=Uint64x2"})
			}
			if cpu.X86.HasPCLMULQDQ {
				// Without PCLMULQDQ, archsimd uses software.
				settings = append(settings, godebugSetting{godebug: size + "cpu.pclmulqdq=off", wantConfig: want + " CLMUL=software"})
			}
		}
	case "arm64":
		// Without PMULL, archsimd emulates carryless multiply using
		// the 8-bit PMULL on Uint8x16. If PMULL is already off, because
		// the CPU lacks it or because GODEBUG turns it off, the
		// current setting tests that.
		if cpu.ARM64.HasPMULL {
			settings = append(settings, godebugSetting{
				godebug:    prefix + "cpu.pmull=off",
				wantConfig: "Emulated=false CLMUL=Uint8x16",
			})
		}
	}
	return settings
}

// cpuOptionsWork reports whether GODEBUG=cpu.X=off settings take effect on
// this operating system. The runtime only applies them where it can read
// the environment early enough in startup, and ignores them elsewhere, such
// as on Windows. See getGodebugEarly in package runtime.
func cpuOptionsWork() bool {
	switch runtime.GOOS {
	case "aix", "darwin", "ios", "dragonfly", "freebsd", "netbsd", "openbsd", "illumos", "solaris", "linux":
		return true
	}
	return false
}

// config returns how the simd package is configured in this process.
func config() string {
	return fmt.Sprintf("VectorBits=%d Emulated=%t CLMUL=%s",
		simd.VectorBits(), simd.Emulated(), simd.InternalCLMULImpl())
}

// TestGODEBUGSettings checks that each of the godebugSettings configures the
// simd package the way it is meant to, so that running the tests under it
// tests what it is meant to.
func TestGODEBUGSettings(t *testing.T) {
	settings := godebugSettings()
	if len(settings) == 1 {
		t.Log("simd is emulated, so there are no other implementations to select")
		return
	}
	exe := testenv.Executable(t)
	for _, s := range settings {
		if s.current {
			continue
		}
		t.Run(s.godebug, func(t *testing.T) {
			cmd := testenv.Command(t, exe, "-print-config")
			cmd.Env = append(cmd.Environ(), "GODEBUG="+s.godebug)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("with GODEBUG=%s: %v\n%s", s.godebug, err, out)
			}
			// Check that the fields in s.wantConfig are a subset
			// of what -print-config printed.
			got := strings.Fields(string(out))
			for _, want := range strings.Fields(s.wantConfig) {
				if !slices.Contains(got, want) {
					t.Errorf("with GODEBUG=%s, simd is configured as %q, want %s", s.godebug, strings.TrimSpace(string(out)), want)
				}
			}
		})
	}
}
