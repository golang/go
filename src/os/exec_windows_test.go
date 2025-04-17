// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package os_test

import (
	"internal/testenv"
	"io"
	. "os"
	"path/filepath"
	"sync"
	"testing"
)

func TestRemoveAllWithExecutedProcess(t *testing.T) {
	// Regression test for golang.org/issue/25965.
	if testing.Short() {
		t.Skip("slow test; skipping")
	}
	testenv.MustHaveExec(t)

	name, err := Executable()
	if err != nil {
		t.Fatal(err)
	}
	r, err := Open(name)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	const n = 100
	var execs [n]string
	// First create n executables.
	for i := 0; i < n; i++ {
		// Rewind r.
		if _, err := r.Seek(0, io.SeekStart); err != nil {
			t.Fatal(err)
		}
		name := filepath.Join(t.TempDir(), "test.exe")
		execs[i] = name
		w, err := Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = io.Copy(w, r); err != nil {
			w.Close()
			t.Fatal(err)
		}
		if err := w.Sync(); err != nil {
			w.Close()
			t.Fatal(err)
		}
		if err = w.Close(); err != nil {
			t.Fatal(err)
		}
	}
	// Then run each executable and remove its directory.
	// Run each executable in a separate goroutine to add some load
	// and increase the chance of triggering the bug.
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			name := execs[i]
			dir := filepath.Dir(name)
			// Run test.exe without executing any test, just to make it do something.
			cmd := testenv.Command(t, name, "-test.run=^$")
			if err := cmd.Run(); err != nil {
				t.Errorf("exec failed: %v", err)
			}
			// Remove dir and check that it doesn't return `ERROR_ACCESS_DENIED`.
			err = RemoveAll(dir)
			if err != nil {
				t.Errorf("RemoveAll failed: %v", err)
			}
		}(i)
	}
	wg.Wait()
}
