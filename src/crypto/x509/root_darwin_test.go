// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestSecPolicyCreateSSLMissingExecutableDir covers go.dev/issue/68557:
// on macOS, SecPolicyCreateSSL returns NULL when the directory containing
// the running executable has been removed (as happens when a process started
// via "go run" outlives the go command's temporary build directory).
func TestSecPolicyCreateSSLMissingExecutableDir(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS_MISSING_EXE_DIR") == "1" {
		missingExecutableDirHelper()
		return
	}

	testenv.MustHaveExec(t)

	exe := testenv.Executable(t)
	dir, err := os.MkdirTemp("", "x509-missing-exe-dir-*")
	if err != nil {
		t.Fatal(err)
	}
	// Intentionally not using t.TempDir: we remove this directory while the
	// helper is still running.
	defer os.RemoveAll(dir)

	helper := filepath.Join(dir, "helper")
	// Prefer a symlink so we do not copy a large test binary (expensive and
	// can get the helper killed under memory pressure during "go test std").
	if err := os.Symlink(exe, helper); err != nil {
		if err := copyFile(helper, exe); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(helper, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	cmd := exec.Command(helper, "-test.run=^TestSecPolicyCreateSSLMissingExecutableDir$")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS_MISSING_EXE_DIR=1")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	// Give the helper time to start, then remove its executable directory.
	time.Sleep(200 * time.Millisecond)
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}

	if err := cmd.Wait(); err != nil {
		t.Fatalf("helper failed: %v\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "OK") {
		t.Fatalf("helper did not report OK\nstdout: %s\nstderr: %s", stdout.String(), stderr.String())
	}
	// repairExecutableDir must not leave the temporary executable directory behind.
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		t.Fatalf("repair left behind executable directory %q", dir)
	}
}

func missingExecutableDirHelper() {
	// Wait for the parent to remove our executable directory.
	deadline := time.Now().Add(5 * time.Second)
	for {
		exe, err := os.Executable()
		if err == nil {
			if _, err := os.Stat(filepath.Dir(exe)); err != nil {
				break
			}
		}
		if time.Now().After(deadline) {
			fmt.Fprintln(os.Stderr, "timeout waiting for executable directory to disappear")
			os.Exit(1)
		}
		time.Sleep(50 * time.Millisecond)
	}

	leaf, err := certificateFromPEM(googleLeaf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse cert: %v\n", err)
		os.Exit(1)
	}
	_, err = leaf.Verify(VerifyOptions{DNSName: "www.google.com"})
	if err != nil && strings.Contains(err.Error(), "SecPolicyCreateSSL") {
		fmt.Fprintf(os.Stderr, "Verify: %v\n", err)
		os.Exit(1)
	}
	// Success means we got past SecPolicyCreateSSL. Verification may still
	// fail for unrelated reasons (expired test cert, etc.).
	fmt.Println("OK")
	os.Exit(0)
}

func copyFile(dst, src string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
