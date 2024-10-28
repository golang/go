// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"crypto/internal/boring"
	"crypto/internal/impl"
	"internal/goos"
	"internal/testenv"
	"testing"
)

// TestAllImplementations runs the provided test function with each available
// implementation of the package registered with crypto/internal/impl. If there
// are no alternative implementations for pkg, f is invoked directly once.
func TestAllImplementations(t *testing.T, pkg string, f func(t *testing.T)) {
	// BoringCrypto bypasses the multiple Go implementations.
	if boring.Enabled {
		f(t)
		return
	}

	impls := impl.List(pkg)
	if len(impls) == 0 {
		f(t)
		return
	}

	t.Cleanup(func() { impl.Reset(pkg) })

	for _, name := range impls {
		if available := impl.Select(pkg, name); available {
			t.Run(name, f)
		} else {
			t.Run(name, func(t *testing.T) {
				// Report an error if we're on Linux CI (assumed to be the most
				// consistent) and the builder can't test this implementation.
				if testenv.Builder() != "" && goos.GOOS == "linux" {
					if name == "SHA-NI" {
						t.Skip("known issue, see golang.org/issue/69592")
					}
					if name == "Armv8.2" {
						t.Skip("known issue, see golang.org/issue/69593")
					}
					t.Error("builder doesn't support CPU features needed to test this implementation")
				} else {
					t.Skip("implementation not supported")
				}
			})
		}

	}

	// Test the generic implementation.
	impl.Select(pkg, "")
	t.Run("Base", f)
}
