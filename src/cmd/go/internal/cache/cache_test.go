// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"fmt"
	"internal/binary"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func init() {
	verify = false // even if GODEBUG is set
}

func TestBasic(t *testing.T) {
	dir, err := os.MkdirTemp("", "cachetest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	_, err = Open(filepath.Join(dir, "notexist"))
	if err == nil {
		t.Fatal(`Open("tmp/notexist") succeeded, want failure`)
	}

	cdir := filepath.Join(dir, "c1")
	if err := os.Mkdir(cdir, 0777); err != nil {
		t.Fatal(err)
	}

	c1, err := Open(cdir)
	if err != nil {
		t.Fatalf("Open(c1) (create): %v", err)
	}
	if err := c1.putIndexEntry(dummyID(1), dummyID(12), 13, true); err != nil {
		t.Fatalf("addIndexEntry: %v", err)
	}
	if err := c1.putIndexEntry(dummyID(1), dummyID(2), 3, true); err != nil { // overwrite entry
		t.Fatalf("addIndexEntry: %v", err)
	}
	if entry, err := c1.Get(dummyID(1)); err != nil || entry.OutputID != dummyID(2) || entry.Size != 3 {
		t.Fatalf("c1.Get(1) = %x, %v, %v, want %x, %v, nil", entry.OutputID, entry.Size, err, dummyID(2), 3)
	}

	c2, err := Open(cdir)
	if err != nil {
		t.Fatalf("Open(c2) (reuse): %v", err)
	}
	if entry, err := c2.Get(dummyID(1)); err != nil || entry.OutputID != dummyID(2) || entry.Size != 3 {
		t.Fatalf("c2.Get(1) = %x, %v, %v, want %x, %v, nil", entry.OutputID, entry.Size, err, dummyID(2), 3)
	}
	if err := c2.putIndexEntry(dummyID(2), dummyID(3), 4, true); err != nil {
		t.Fatalf("addIndexEntry: %v", err)
	}
	if entry, err := c1.Get(dummyID(2)); err != nil || entry.OutputID != dummyID(3) || entry.Size != 4 {
		t.Fatalf("c1.Get(2) = %x, %v, %v, want %x, %v, nil", entry.OutputID, entry.Size, err, dummyID(3), 4)
	}
}

func TestGrowth(t *testing.T) {
	dir, err := os.MkdirTemp("", "cachetest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	c, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	n := 10000
	if testing.Short() {
		n = 10
	}

	for i := 0; i < n; i++ {
		if err := c.putIndexEntry(dummyID(i), dummyID(i*99), int64(i)*101, true); err != nil {
			t.Fatalf("addIndexEntry: %v", err)
		}
		id := ActionID(dummyID(i))
		entry, err := c.Get(id)
		if err != nil {
			t.Fatalf("Get(%x): %v", id, err)
		}
		if entry.OutputID != dummyID(i*99) || entry.Size != int64(i)*101 {
			t.Errorf("Get(%x) = %x, %d, want %x, %d", id, entry.OutputID, entry.Size, dummyID(i*99), int64(i)*101)
		}
	}
	for i := 0; i < n; i++ {
		id := ActionID(dummyID(i))
		entry, err := c.Get(id)
		if err != nil {
			t.Fatalf("Get2(%x): %v", id, err)
		}
		if entry.OutputID != dummyID(i*99) || entry.Size != int64(i)*101 {
			t.Errorf("Get2(%x) = %x, %d, want %x, %d", id, entry.OutputID, entry.Size, dummyID(i*99), int64(i)*101)
		}
	}
}

func TestVerifyPanic(t *testing.T) {
	os.Setenv("GODEBUG", "gocacheverify=1")
	initEnv()
	defer func() {
		os.Unsetenv("GODEBUG")
		verify = false
	}()

	if !verify {
		t.Fatal("initEnv did not set verify")
	}

	dir, err := os.MkdirTemp("", "cachetest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	c, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	id := ActionID(dummyID(1))
	if err := PutBytes(c, id, []byte("abc")); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if err := recover(); err != nil {
			t.Log(err)
			return
		}
	}()
	PutBytes(c, id, []byte("def"))
	t.Fatal("mismatched Put did not panic in verify mode")
}

func dummyID(x int) [HashSize]byte {
	var out [HashSize]byte
	binary.LittleEndian.PutUint64(out[:], uint64(x))
	return out
}

func TestCacheTrim(t *testing.T) {
	dir, err := os.MkdirTemp("", "cachetest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	c, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	const start = 1000000000
	now := int64(start)
	c.now = func() time.Time { return time.Unix(now, 0) }

	checkTime := func(name string, mtime int64) {
		t.Helper()
		file := filepath.Join(c.dir, name[:2], name)
		info, err := os.Stat(file)
		if err != nil {
			t.Fatal(err)
		}
		if info.ModTime().Unix() != mtime {
			t.Fatalf("%s mtime = %d, want %d", name, info.ModTime().Unix(), mtime)
		}
	}

	id := ActionID(dummyID(1))
	PutBytes(c, id, []byte("abc"))
	entry, _ := c.Get(id)
	PutBytes(c, ActionID(dummyID(2)), []byte("def"))
	mtime := now
	checkTime(fmt.Sprintf("%x-a", id), mtime)
	checkTime(fmt.Sprintf("%x-d", entry.OutputID), mtime)

	// Get should not change recent mtimes.
	now = start + 10
	c.Get(id)
	checkTime(fmt.Sprintf("%x-a", id), mtime)
	checkTime(fmt.Sprintf("%x-d", entry.OutputID), mtime)

	// Get should change distant mtimes.
	now = start + 5000
	mtime2 := now
	if _, err := c.Get(id); err != nil {
		t.Fatal(err)
	}
	c.OutputFile(entry.OutputID)
	checkTime(fmt.Sprintf("%x-a", id), mtime2)
	checkTime(fmt.Sprintf("%x-d", entry.OutputID), mtime2)

	// Trim should leave everything alone: it's all too new.
	if err := c.Trim(); err != nil {
		if testenv.SyscallIsNotSupported(err) {
			t.Skipf("skipping: Trim is unsupported (%v)", err)
		}
		t.Fatal(err)
	}
	if _, err := c.Get(id); err != nil {
		t.Fatal(err)
	}
	c.OutputFile(entry.OutputID)
	data, err := os.ReadFile(filepath.Join(dir, "trim.txt"))
	if err != nil {
		t.Fatal(err)
	}
	checkTime(fmt.Sprintf("%x-a", dummyID(2)), start)

	// Trim less than a day later should not do any work at all.
	now = start + 80000
	if err := c.Trim(); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Get(id); err != nil {
		t.Fatal(err)
	}
	c.OutputFile(entry.OutputID)
	data2, err := os.ReadFile(filepath.Join(dir, "trim.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, data2) {
		t.Fatalf("second trim did work: %q -> %q", data, data2)
	}

	// Fast forward and do another trim just before the 5 day cutoff.
	// Note that because of usedQuantum the cutoff is actually 5 days + 1 hour.
	// We used c.Get(id) just now, so 5 days later it should still be kept.
	// On the other hand almost a full day has gone by since we wrote dummyID(2)
	// and we haven't looked at it since, so 5 days later it should be gone.
	now += 5 * 86400
	checkTime(fmt.Sprintf("%x-a", dummyID(2)), start)
	if err := c.Trim(); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Get(id); err != nil {
		t.Fatal(err)
	}
	c.OutputFile(entry.OutputID)
	mtime3 := now
	if _, err := c.Get(dummyID(2)); err == nil { // haven't done a Get for this since original write above
		t.Fatalf("Trim did not remove dummyID(2)")
	}

	// The c.Get(id) refreshed id's mtime again.
	// Check that another 5 days later it is still not gone,
	// but check by using checkTime, which doesn't bring mtime forward.
	now += 5 * 86400
	if err := c.Trim(); err != nil {
		t.Fatal(err)
	}
	checkTime(fmt.Sprintf("%x-a", id), mtime3)
	checkTime(fmt.Sprintf("%x-d", entry.OutputID), mtime3)

	// Half a day later Trim should still be a no-op, because there was a Trim recently.
	// Even though the entry for id is now old enough to be trimmed,
	// it gets a reprieve until the time comes for a new Trim scan.
	now += 86400 / 2
	if err := c.Trim(); err != nil {
		t.Fatal(err)
	}
	checkTime(fmt.Sprintf("%x-a", id), mtime3)
	checkTime(fmt.Sprintf("%x-d", entry.OutputID), mtime3)

	// Another half a day later, Trim should actually run, and it should remove id.
	now += 86400/2 + 1
	if err := c.Trim(); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Get(dummyID(1)); err == nil {
		t.Fatal("Trim did not remove dummyID(1)")
	}
}
