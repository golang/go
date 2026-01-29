// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"crypto/internal/boring"
	"crypto/internal/impl"
	"internal/goarch"
	"internal/goos"
	"internal/testenv"
	"strings"
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
				// Report an error if we're on the most capable builder for the
				// architecture and the builder can't test this implementation.
				if flagshipBuilder() {
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

func flagshipBuilder() bool {
	builder := testenv.Builder()
	if builder == "" {
		return false
	}
	switch goarch.GOARCH {
	case "amd64":
		return strings.Contains(builder, "_avx512")
	case "arm64":
		// Apple M chips support everything we use.
		return goos.GOOS == "darwin"
	default:
		// Presumably the Linux builders are the most capable.
		return goos.GOOS == "linux"
	}
}
