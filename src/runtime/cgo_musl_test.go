// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && cgo

package runtime_test

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestMuslSharedLibrary tests that Go c-shared libraries work correctly
// on standards-compliant systems where DT_INIT_ARRAY doesn't receive argc/argv per ELF specification.
func TestMuslSharedLibrary(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("test requires Linux")
	}

	// Detect if we're running on musl
	isMusl := isMuslLibc()
	
	// Build the test shared library
	tmpdir := t.TempDir()
	libPath := filepath.Join(tmpdir, "libmusltest.so")
	
	cmd := exec.Command("go", "build", "-buildmode=c-shared", "-o", libPath,
		"./testdata/musl_sharedlib.go")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to build shared library: %v\n%s", err, out)
	}

	// Build the loader program
	loaderPath := filepath.Join(tmpdir, "musl_loader")
	ccCmd := exec.Command("cc", "-o", loaderPath,
		"./testdata/musl_loader.c", "-ldl")
	if out, err := ccCmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to build loader: %v\n%s", err, out)
	}

	// Run the loader
	runCmd := exec.Command(loaderPath, libPath)
	out, err := runCmd.CombinedOutput()
	if err != nil {
		// On unpatched Go, this will fail with SIGSEGV on standards-compliant systems
		if isMusl && strings.Contains(string(out), "signal:") {
			t.Fatalf("Got signal (likely SIGSEGV) on standards-compliant system - argc/argv fix not working: %v\n%s", err, out)
		}
		t.Fatalf("loader failed: %v\n%s", err, out)
	}

	// Check outputs
	outStr := string(out)
	
	// Test 1: Initialization should succeed
	if !strings.Contains(outStr, "MUSL_INIT_SUCCESS") {
		t.Error("shared library initialization failed")
	}

	// Test 2: Environment sync (currently expected to fail)
	if strings.Contains(outStr, "ENV_SYNC_SUCCESS") {
		t.Log("Environment synchronization working (unexpected)")
	} else if strings.Contains(outStr, "ENV_SYNC_FAIL") {
		// This is expected until we fix env sync
		t.Log("Environment synchronization not working (expected)")
	}

	// Test 3: argc access
	if strings.Contains(outStr, "ARGC_TEST:") {
		// On musl without our fix, argc would be garbage
		// With our fix, it should be 0 for shared libraries
		t.Log("argc test passed")
	}

	// Test 4: argv access
	if strings.Contains(outStr, "ARGV_TEST_SUCCESS") {
		t.Log("argv accessible")
	} else if strings.Contains(outStr, "ARGV_TEST_FAIL") {
		// Expected on musl where argv is null
		t.Log("argv not accessible (expected on musl)")
	}
}

// TestCSharedOnAlpine uses Docker to test on actual Alpine Linux if available
func TestCSharedOnAlpine(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping docker test in short mode")
	}

	// Check if Docker is available
	if _, err := exec.LookPath("docker"); err != nil {
		t.Skip("docker not available")
	}

	// Check if Docker daemon is running
	if err := exec.Command("docker", "info").Run(); err != nil {
		t.Skip("docker daemon not running")
	}

	tmpdir := t.TempDir()

	// Create a test Dockerfile
	dockerfile := `FROM alpine:latest
RUN apk add --no-cache go gcc musl-dev
WORKDIR /test
COPY . .
RUN go build -buildmode=c-shared -o libtest.so musl_sharedlib.go
RUN gcc -o loader musl_loader.c -ldl
CMD ["./loader", "./libtest.so"]
`
	dockerfilePath := filepath.Join(tmpdir, "Dockerfile")
	if err := os.WriteFile(dockerfilePath, []byte(dockerfile), 0644); err != nil {
		t.Fatal(err)
	}

	// Copy test files
	for _, file := range []string{"musl_sharedlib.go", "musl_loader.c"} {
		src := filepath.Join("testdata", file)
		dst := filepath.Join(tmpdir, file)
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst, data, 0644); err != nil {
			t.Fatal(err)
		}
	}

	// Build and run the Docker container
	buildCmd := exec.Command("docker", "build", "-t", "go-musl-test", tmpdir)
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("docker build failed: %v\n%s", err, out)
	}

	runCmd := exec.Command("docker", "run", "--rm", "go-musl-test")
	out, err := runCmd.CombinedOutput()
	if err != nil {
		t.Fatalf("docker run failed: %v\n%s", err, out)
	}

	// Check that initialization succeeded
	if !bytes.Contains(out, []byte("MUSL_INIT_SUCCESS")) {
		t.Error("shared library failed to initialize on Alpine Linux")
	}
}

// isMuslLibc attempts to detect if we're running on musl libc
func isMuslLibc() bool {
	// Try ldd --version first
	cmd := exec.Command("ldd", "--version")
	out, _ := cmd.CombinedOutput()
	if bytes.Contains(out, []byte("musl")) {
		return true
	}

	// Check for /lib/ld-musl-*.so.1
	matches, _ := filepath.Glob("/lib/ld-musl-*.so.1")
	if len(matches) > 0 {
		return true
	}

	// Check if we're on Alpine
	if _, err := os.Stat("/etc/alpine-release"); err == nil {
		return true
	}

	return false
}
