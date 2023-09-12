// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

func TestAcquireFileLock(t *testing.T) {
	name := filepath.Join(t.TempDir(), "config.json")

	const concurrency = 100
	var acquired int32
	var releasers [concurrency]func()
	defer func() {
		for _, r := range releasers {
			if r != nil {
				r()
			}
		}
	}()

	var wg sync.WaitGroup
	for i := range releasers {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()

			release, ok, err := acquireLockFile(name)
			if err != nil {
				t.Errorf("Acquire failed: %v", err)
				return
			}
			if ok {
				atomic.AddInt32(&acquired, 1)
				releasers[i] = release
			}
		}()
	}

	wg.Wait()

	if acquired != 1 {
		t.Errorf("Acquire succeeded %d times, expected exactly 1", acquired)
	}
}

func TestReleaseAndAcquireFileLock(t *testing.T) {
	name := filepath.Join(t.TempDir(), "config.json")

	acquire := func() (func(), bool) {
		t.Helper()
		release, ok, err := acquireLockFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return release, ok
	}

	release, ok := acquire()
	if !ok {
		t.Fatal("failed to Acquire")
	}
	if release2, ok := acquire(); ok {
		release()
		release2()
		t.Fatalf("Acquire succeeded unexpectedly")
	}

	release()
	release3, ok := acquire()
	release3()
	if !ok {
		t.Fatalf("failed to Acquire")
	}
}
