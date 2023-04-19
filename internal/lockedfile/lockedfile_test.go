// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || plan9 || windows
// +build unix aix darwin dragonfly freebsd linux netbsd openbsd solaris plan9 windows

package lockedfile_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/tools/internal/lockedfile"
)

func mustTempDir(t *testing.T) (dir string, remove func()) {
	t.Helper()

	dir, err := os.MkdirTemp("", filepath.Base(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	return dir, func() { os.RemoveAll(dir) }
}

const (
	quiescent            = 10 * time.Millisecond
	probablyStillBlocked = 10 * time.Second
)

func mustBlock(t *testing.T, desc string, f func()) (wait func(*testing.T)) {
	t.Helper()

	done := make(chan struct{})
	go func() {
		f()
		close(done)
	}()

	select {
	case <-done:
		t.Fatalf("%s unexpectedly did not block", desc)
		return nil

	case <-time.After(quiescent):
		return func(t *testing.T) {
			t.Helper()
			select {
			case <-time.After(probablyStillBlocked):
				t.Fatalf("%s is unexpectedly still blocked after %v", desc, probablyStillBlocked)
			case <-done:
			}
		}
	}
}

func TestMutexExcludes(t *testing.T) {
	t.Parallel()

	dir, remove := mustTempDir(t)
	defer remove()

	path := filepath.Join(dir, "lock")

	mu := lockedfile.MutexAt(path)
	t.Logf("mu := MutexAt(_)")

	unlock, err := mu.Lock()
	if err != nil {
		t.Fatalf("mu.Lock: %v", err)
	}
	t.Logf("unlock, _  := mu.Lock()")

	mu2 := lockedfile.MutexAt(mu.Path)
	t.Logf("mu2 := MutexAt(mu.Path)")

	wait := mustBlock(t, "mu2.Lock()", func() {
		unlock2, err := mu2.Lock()
		if err != nil {
			t.Errorf("mu2.Lock: %v", err)
			return
		}
		t.Logf("unlock2, _ := mu2.Lock()")
		t.Logf("unlock2()")
		unlock2()
	})

	t.Logf("unlock()")
	unlock()
	wait(t)
}

func TestReadWaitsForLock(t *testing.T) {
	t.Parallel()

	dir, remove := mustTempDir(t)
	defer remove()

	path := filepath.Join(dir, "timestamp.txt")

	f, err := lockedfile.Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer f.Close()

	const (
		part1 = "part 1\n"
		part2 = "part 2\n"
	)
	_, err = f.WriteString(part1)
	if err != nil {
		t.Fatalf("WriteString: %v", err)
	}
	t.Logf("WriteString(%q) = <nil>", part1)

	wait := mustBlock(t, "Read", func() {
		b, err := lockedfile.Read(path)
		if err != nil {
			t.Errorf("Read: %v", err)
			return
		}

		const want = part1 + part2
		got := string(b)
		if got == want {
			t.Logf("Read(_) = %q", got)
		} else {
			t.Errorf("Read(_) = %q, _; want %q", got, want)
		}
	})

	_, err = f.WriteString(part2)
	if err != nil {
		t.Errorf("WriteString: %v", err)
	} else {
		t.Logf("WriteString(%q) = <nil>", part2)
	}
	f.Close()

	wait(t)
}

func TestCanLockExistingFile(t *testing.T) {
	t.Parallel()

	dir, remove := mustTempDir(t)
	defer remove()
	path := filepath.Join(dir, "existing.txt")

	if err := os.WriteFile(path, []byte("ok"), 0777); err != nil {
		t.Fatalf("os.WriteFile: %v", err)
	}

	f, err := lockedfile.Edit(path)
	if err != nil {
		t.Fatalf("first Edit: %v", err)
	}

	wait := mustBlock(t, "Edit", func() {
		other, err := lockedfile.Edit(path)
		if err != nil {
			t.Errorf("second Edit: %v", err)
		}
		other.Close()
	})

	f.Close()
	wait(t)
}

// TestSpuriousEDEADLK verifies that the spurious EDEADLK reported in
// https://golang.org/issue/32817 no longer occurs.
func TestSpuriousEDEADLK(t *testing.T) {
	// 	P.1 locks file A.
	// 	Q.3 locks file B.
	// 	Q.3 blocks on file A.
	// 	P.2 blocks on file B. (Spurious EDEADLK occurs here.)
	// 	P.1 unlocks file A.
	// 	Q.3 unblocks and locks file A.
	// 	Q.3 unlocks files A and B.
	// 	P.2 unblocks and locks file B.
	// 	P.2 unlocks file B.

	dirVar := t.Name() + "DIR"

	if dir := os.Getenv(dirVar); dir != "" {
		// Q.3 locks file B.
		b, err := lockedfile.Edit(filepath.Join(dir, "B"))
		if err != nil {
			t.Fatal(err)
		}
		defer b.Close()

		if err := os.WriteFile(filepath.Join(dir, "locked"), []byte("ok"), 0666); err != nil {
			t.Fatal(err)
		}

		// Q.3 blocks on file A.
		a, err := lockedfile.Edit(filepath.Join(dir, "A"))
		// Q.3 unblocks and locks file A.
		if err != nil {
			t.Fatal(err)
		}
		defer a.Close()

		// Q.3 unlocks files A and B.
		return
	}

	dir, remove := mustTempDir(t)
	defer remove()

	// P.1 locks file A.
	a, err := lockedfile.Edit(filepath.Join(dir, "A"))
	if err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(os.Args[0], "-test.run="+t.Name())
	cmd.Env = append(os.Environ(), fmt.Sprintf("%s=%s", dirVar, dir))

	qDone := make(chan struct{})
	waitQ := mustBlock(t, "Edit A and B in subprocess", func() {
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("%v:\n%s", err, out)
		}
		close(qDone)
	})

	// Wait until process Q has either failed or locked file B.
	// Otherwise, P.2 might not block on file B as intended.
locked:
	for {
		if _, err := os.Stat(filepath.Join(dir, "locked")); !os.IsNotExist(err) {
			break locked
		}
		select {
		case <-qDone:
			break locked
		case <-time.After(1 * time.Millisecond):
		}
	}

	waitP2 := mustBlock(t, "Edit B", func() {
		// P.2 blocks on file B. (Spurious EDEADLK occurs here.)
		b, err := lockedfile.Edit(filepath.Join(dir, "B"))
		// P.2 unblocks and locks file B.
		if err != nil {
			t.Error(err)
			return
		}
		// P.2 unlocks file B.
		b.Close()
	})

	// P.1 unlocks file A.
	a.Close()

	waitQ(t)
	waitP2(t)
}
