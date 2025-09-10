// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || plan9 || wasip1

package syscall_test

import (
	"fmt"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

type env struct {
	name, val string
}

func genDummyEnv(tb testing.TB, size int) []env {
	tb.Helper()
	envList := make([]env, size)
	for idx := range size {
		envList[idx] = env{
			name: fmt.Sprintf("DUMMY_VAR_%d", idx),
			val:  fmt.Sprintf("val-%d", idx*100),
		}
	}
	return envList
}

func setDummyEnv(tb testing.TB, envList []env) {
	tb.Helper()
	for _, env := range envList {
		if err := syscall.Setenv(env.name, env.val); err != nil {
			tb.Fatalf("setenv %s=%q failed: %v", env.name, env.val, err)
		}
	}
}

func setupEnvCleanup(tb testing.TB) {
	tb.Helper()
	originalEnv := map[string]string{}
	for _, env := range syscall.Environ() {
		fields := strings.SplitN(env, "=", 2)
		name, val := fields[0], fields[1]
		originalEnv[name] = val
	}
	tb.Cleanup(func() {
		syscall.Clearenv()
		for name, val := range originalEnv {
			if err := syscall.Setenv(name, val); err != nil {
				tb.Fatalf("could not reset env %s=%q: %v", name, val, err)
			}
		}
	})
}

func TestClearenv(t *testing.T) {
	setupEnvCleanup(t)

	t.Run("DummyVars-4096", func(t *testing.T) {
		envList := genDummyEnv(t, 4096)
		setDummyEnv(t, envList)

		if env := syscall.Environ(); len(env) < 4096 {
			t.Fatalf("env is missing dummy variables: %v", env)
		}
		for idx := range 4096 {
			name := fmt.Sprintf("DUMMY_VAR_%d", idx)
			if _, ok := syscall.Getenv(name); !ok {
				t.Fatalf("env is missing dummy variable %s", name)
			}
		}

		syscall.Clearenv()

		if env := syscall.Environ(); len(env) != 0 {
			t.Fatalf("clearenv should've cleared all variables: %v still set", env)
		}
		for idx := range 4096 {
			name := fmt.Sprintf("DUMMY_VAR_%d", idx)
			if val, ok := syscall.Getenv(name); ok {
				t.Fatalf("clearenv should've cleared all variables: %s=%q still set", name, val)
			}
		}
	})

	// Test that GODEBUG getting cleared by Clearenv also resets the behaviour.
	t.Run("GODEBUG", func(t *testing.T) {
		envList := genDummyEnv(t, 100)
		setDummyEnv(t, envList)

		doNilPanic := func() (ret any) {
			defer func() {
				ret = recover()
			}()
			if true { // defeat vet's unreachable pass
				panic(nil)
			}
			return "should not return"
		}

		// Allow panic(nil).
		if err := syscall.Setenv("GODEBUG", "panicnil=1"); err != nil {
			t.Fatalf("setenv GODEBUG=panicnil=1 failed: %v", err)
		}

		got := doNilPanic()
		if got != nil {
			t.Fatalf("GODEBUG=panicnil=1 did not allow for nil panic: got %#v", got)
		}

		// Disallow panic(nil).
		syscall.Clearenv()

		if env := syscall.Environ(); len(env) != 0 {
			t.Fatalf("clearenv should've cleared all variables: %v still set", env)
		}

		got = doNilPanic()
		if got == nil {
			t.Fatalf("GODEBUG=panicnil=1 being unset didn't reset panicnil behaviour")
		}
		if godebug, ok := syscall.Getenv("GODEBUG"); ok {
			t.Fatalf("GODEBUG still exists in environment despite being unset: GODEBUG=%q", godebug)
		}
	})
}

func BenchmarkClearenv(b *testing.B) {
	setupEnvCleanup(b)
	b.ResetTimer()
	for _, size := range []int{100, 1000, 10000} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			envList := genDummyEnv(b, size)
			for b.Loop() {
				// Ideally we would use b.StopTimer() for the setDummyEnv
				// portion, but this causes the benchmark time to get confused
				// and take forever. See <https://go.dev/issue/27217>.
				setDummyEnv(b, envList)
				syscall.Clearenv()
			}
		})
	}
}
