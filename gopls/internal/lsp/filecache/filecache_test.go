// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filecache_test

// This file defines tests of the API of the filecache package.
//
// Some properties (e.g. garbage collection) cannot be exercised
// through the API, so this test does not attempt to do so.

import (
	"bytes"
	cryptorand "crypto/rand"
	"fmt"
	"log"
	mathrand "math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/gopls/internal/lsp/filecache"
	"golang.org/x/tools/internal/testenv"
)

func TestBasics(t *testing.T) {
	const kind = "TestBasics"
	key := uniqueKey() // never used before
	value := []byte("hello")

	// Get of a never-seen key returns not found.
	if _, err := filecache.Get(kind, key); err != filecache.ErrNotFound {
		if strings.Contains(err.Error(), "operation not supported") ||
			strings.Contains(err.Error(), "not implemented") {
			t.Skipf("skipping: %v", err)
		}
		t.Errorf("Get of random key returned err=%q, want not found", err)
	}

	// Set of a never-seen key and a small value succeeds.
	if err := filecache.Set(kind, key, value); err != nil {
		t.Errorf("Set failed: %v", err)
	}

	// Get of the key returns a copy of the value.
	if got, err := filecache.Get(kind, key); err != nil {
		t.Errorf("Get after Set failed: %v", err)
	} else if string(got) != string(value) {
		t.Errorf("Get after Set returned different value: got %q, want %q", got, value)
	}

	// The kind is effectively part of the key.
	if _, err := filecache.Get("different-kind", key); err != filecache.ErrNotFound {
		t.Errorf("Get with wrong kind returned err=%q, want not found", err)
	}
}

// TestConcurrency exercises concurrent access to the same entry.
func TestConcurrency(t *testing.T) {
	if os.Getenv("GO_BUILDER_NAME") == "plan9-arm" {
		t.Skip(`skipping on plan9-arm builder due to golang/go#58748: failing with 'mount rpc error'`)
	}
	const kind = "TestConcurrency"
	key := uniqueKey()
	const N = 100 // concurrency level

	// Construct N distinct values, each larger
	// than a typical 4KB OS file buffer page.
	var values [N][8192]byte
	for i := range values {
		if _, err := mathrand.Read(values[i][:]); err != nil {
			t.Fatalf("rand: %v", err)
		}
	}

	// get calls Get and verifies that the cache entry
	// matches one of the values passed to Set.
	get := func(mustBeFound bool) error {
		got, err := filecache.Get(kind, key)
		if err != nil {
			if err == filecache.ErrNotFound && !mustBeFound {
				return nil // not found
			}
			return err
		}
		for _, want := range values {
			if bytes.Equal(want[:], got) {
				return nil // a match
			}
		}
		return fmt.Errorf("Get returned a value that was never Set")
	}

	// Perform N concurrent calls to Set and Get.
	// All sets must succeed.
	// All gets must return nothing, or one of the Set values;
	// there is no third possibility.
	var group errgroup.Group
	for i := range values {
		i := i
		group.Go(func() error { return filecache.Set(kind, key, values[i][:]) })
		group.Go(func() error { return get(false) })
	}
	if err := group.Wait(); err != nil {
		if strings.Contains(err.Error(), "operation not supported") ||
			strings.Contains(err.Error(), "not implemented") {
			t.Skipf("skipping: %v", err)
		}
		t.Fatal(err)
	}

	// A final Get must report one of the values that was Set.
	if err := get(true); err != nil {
		t.Fatalf("final Get failed: %v", err)
	}
}

const (
	testIPCKind   = "TestIPC"
	testIPCValueA = "hello"
	testIPCValueB = "world"
)

// TestIPC exercises interprocess communication through the cache.
// It calls Set(A) in the parent, { Get(A); Set(B) } in the child
// process, then Get(B) in the parent.
func TestIPC(t *testing.T) {
	testenv.NeedsExec(t)

	keyA := uniqueKey()
	keyB := uniqueKey()
	value := []byte(testIPCValueA)

	// Set keyA.
	if err := filecache.Set(testIPCKind, keyA, value); err != nil {
		if strings.Contains(err.Error(), "operation not supported") {
			t.Skipf("skipping: %v", err)
		}
		t.Fatalf("Set: %v", err)
	}

	// Call ipcChild in a child process,
	// passing it the keys in the environment
	// (quoted, to avoid NUL termination of C strings).
	// It will Get(A) then Set(B).
	cmd := exec.Command(os.Args[0], os.Args[1:]...)
	cmd.Env = append(os.Environ(),
		"ENTRYPOINT=ipcChild",
		fmt.Sprintf("KEYA=%q", keyA),
		fmt.Sprintf("KEYB=%q", keyB))
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}

	// Verify keyB.
	got, err := filecache.Get(testIPCKind, keyB)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "world" {
		t.Fatalf("Get(keyB) = %q, want %q", got, "world")
	}
}

// We define our own main function so that portions of
// some tests can run in a separate (child) process.
func TestMain(m *testing.M) {
	switch os.Getenv("ENTRYPOINT") {
	case "ipcChild":
		ipcChild()
	default:
		os.Exit(m.Run())
	}
}

// ipcChild is the portion of TestIPC that runs in a child process.
func ipcChild() {
	getenv := func(name string) (key [32]byte) {
		s, _ := strconv.Unquote(os.Getenv(name))
		copy(key[:], []byte(s))
		return
	}

	// Verify key A.
	got, err := filecache.Get(testIPCKind, getenv("KEYA"))
	if err != nil || string(got) != testIPCValueA {
		log.Fatalf("child: Get(key) = %q, %v; want %q", got, err, testIPCValueA)
	}

	// Set key B.
	if err := filecache.Set(testIPCKind, getenv("KEYB"), []byte(testIPCValueB)); err != nil {
		log.Fatalf("child: Set(keyB) failed: %v", err)
	}
}

// uniqueKey returns a key that has never been used before.
func uniqueKey() (key [32]byte) {
	if _, err := cryptorand.Read(key[:]); err != nil {
		log.Fatalf("rand: %v", err)
	}
	return
}

func BenchmarkUncontendedGet(b *testing.B) {
	const kind = "BenchmarkUncontendedGet"
	key := uniqueKey()

	var value [8192]byte
	if _, err := mathrand.Read(value[:]); err != nil {
		b.Fatalf("rand: %v", err)
	}
	if err := filecache.Set(kind, key, value[:]); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.SetBytes(int64(len(value)))

	var group errgroup.Group
	group.SetLimit(50)
	for i := 0; i < b.N; i++ {
		group.Go(func() error {
			_, err := filecache.Get(kind, key)
			return err
		})
	}
	if err := group.Wait(); err != nil {
		b.Fatal(err)
	}
}

// These two benchmarks are asymmetric: the one for Get imposes a
// modest bound on concurrency (50) whereas the one for Set imposes a
// much higher concurrency (1000) to test the implementation's
// self-imposed bound.

func BenchmarkUncontendedSet(b *testing.B) {
	const kind = "BenchmarkUncontendedSet"
	key := uniqueKey()
	var value [8192]byte

	const P = 1000 // parallelism
	b.SetBytes(P * int64(len(value)))

	for i := 0; i < b.N; i++ {
		// Perform P concurrent calls to Set. All must succeed.
		var group errgroup.Group
		for range [P]bool{} {
			group.Go(func() error {
				return filecache.Set(kind, key, value[:])
			})
		}
		if err := group.Wait(); err != nil {
			if strings.Contains(err.Error(), "operation not supported") ||
				strings.Contains(err.Error(), "not implemented") {
				b.Skipf("skipping: %v", err)
			}
			b.Fatal(err)
		}
	}
}
