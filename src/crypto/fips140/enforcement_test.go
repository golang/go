// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140_test

import (
	"crypto/des"
	"crypto/fips140"
	"crypto/internal/cryptotest"
	"internal/testenv"
	"testing"
)

func expectAllowed(t *testing.T, why string, expected bool) {
	t.Helper()
	result := isAllowed()
	if result != expected {
		t.Fatalf("%v: expected: %v, got: %v", why, expected, result)
	}
}

func isAllowed() bool {
	_, err := des.NewCipher(make([]byte, 8))
	return err == nil
}

func TestWithoutEnforcement(t *testing.T) {
	cryptotest.MustSupportFIPS140(t)
	if !fips140.Enforced() {
		cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestWithoutEnforcement$", "-test.v")
		cmd.Env = append(cmd.Environ(), "GODEBUG=fips140=only")
		out, err := cmd.CombinedOutput()
		t.Logf("running with GODEBUG=fips140=only:\n%s", out)
		if err != nil {
			t.Errorf("fips140=only subprocess failed: %v", err)
		}
		return
	}

	t.Run("Disabled", func(t *testing.T) {
		expectAllowed(t, "before enforcement disabled", false)
		fips140.WithoutEnforcement(func() {
			expectAllowed(t, "inside WithoutEnforcement", true)
		})
		// make sure that bypass doesn't live on after returning
		expectAllowed(t, "after WithoutEnforcement", false)
	})

	t.Run("Nested", func(t *testing.T) {
		expectAllowed(t, "before enforcement bypass", false)
		fips140.WithoutEnforcement(func() {
			fips140.WithoutEnforcement(func() {
				expectAllowed(t, "inside nested WithoutEnforcement", true)
			})
			expectAllowed(t, "inside nested WithoutEnforcement", true)
		})
		expectAllowed(t, "after enforcement bypass", false)
	})

	t.Run("GoroutineInherit", func(t *testing.T) {
		ch := make(chan bool, 2)
		expectAllowed(t, "before enforcement bypass", false)
		fips140.WithoutEnforcement(func() {
			go func() {
				ch <- isAllowed()
			}()
		})
		allowed := <-ch
		if !allowed {
			t.Fatal("goroutine didn't inherit enforcement bypass")
		}
		go func() {
			ch <- isAllowed()
		}()
		allowed = <-ch
		if allowed {
			t.Fatal("goroutine inherited bypass after WithoutEnforcement return")
		}
	})
}
