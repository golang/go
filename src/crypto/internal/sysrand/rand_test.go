// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sysrand

import (
	"bytes"
	"compress/flate"
	"internal/testenv"
	"os"
	"runtime"
	"sync"
	"testing"
)

func TestRead(t *testing.T) {
	// 40MiB, more than the documented maximum of 32Mi-1 on Linux 32-bit.
	b := make([]byte, 40<<20)
	Read(b)

	if testing.Short() {
		b = b[len(b)-100_000:]
	}

	var z bytes.Buffer
	f, _ := flate.NewWriter(&z, 5)
	f.Write(b)
	f.Close()
	if z.Len() < len(b)*99/100 {
		t.Fatalf("Compressed %d -> %d", len(b), z.Len())
	}
}

func TestReadByteValues(t *testing.T) {
	b := make([]byte, 1)
	v := make(map[byte]bool)
	for {
		Read(b)
		v[b[0]] = true
		if len(v) == 256 {
			break
		}
	}
}

func TestReadEmpty(t *testing.T) {
	Read(make([]byte, 0))
	Read(nil)
}

func TestConcurrentRead(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	const N = 100
	const M = 1000
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			for i := 0; i < M; i++ {
				b := make([]byte, 32)
				Read(b)
			}
		}()
	}
	wg.Wait()
}

// TestNoUrandomFallback ensures the urandom fallback is not reached in
// normal operations.
func TestNoUrandomFallback(t *testing.T) {
	expectFallback := false
	if runtime.GOOS == "aix" {
		// AIX always uses the urandom fallback.
		expectFallback = true
	}
	if os.Getenv("GO_GETRANDOM_DISABLED") == "1" {
		// We are testing the urandom fallback intentionally.
		expectFallback = true
	}
	Read(make([]byte, 1))
	if urandomFile != nil && !expectFallback {
		t.Error("/dev/urandom fallback used unexpectedly")
		t.Log("note: if this test fails, it may be because the system does not have getrandom(2)")
	}
	if urandomFile == nil && expectFallback {
		t.Error("/dev/urandom fallback not used as expected")
	}
}

func TestReadError(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	testenv.MustHaveExec(t)

	// We run this test in a subprocess because it's expected to crash.
	if os.Getenv("GO_TEST_READ_ERROR") == "1" {
		testingOnlyFailRead = true
		Read(make([]byte, 32))
		t.Error("Read did not crash")
		return
	}

	cmd := testenv.Command(t, os.Args[0], "-test.run=TestReadError")
	cmd.Env = append(os.Environ(), "GO_TEST_READ_ERROR=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Error("subprocess succeeded unexpectedly")
	}
	exp := "fatal error: crypto/rand: failed to read random data"
	if !bytes.Contains(out, []byte(exp)) {
		t.Errorf("subprocess output does not contain %q: %s", exp, out)
	}
}
