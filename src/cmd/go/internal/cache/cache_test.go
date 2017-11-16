// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"encoding/binary"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func init() {
	verify = false // even if GODEBUG is set
}

func TestBasic(t *testing.T) {
	dir, err := ioutil.TempDir("", "cachetest-")
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
	dir, err := ioutil.TempDir("", "cachetest-")
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
		n = 1000
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

	dir, err := ioutil.TempDir("", "cachetest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	c, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	id := ActionID(dummyID(1))
	if err := c.PutBytes(id, []byte("abc")); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if err := recover(); err != nil {
			t.Log(err)
			return
		}
	}()
	c.PutBytes(id, []byte("def"))
	t.Fatal("mismatched Put did not panic in verify mode")
}

func TestCacheLog(t *testing.T) {
	dir, err := ioutil.TempDir("", "cachetest-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	c, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	c.now = func() time.Time { return time.Unix(1e9, 0) }

	id := ActionID(dummyID(1))
	c.Get(id)
	c.PutBytes(id, []byte("abc"))
	c.Get(id)

	c, err = Open(dir)
	if err != nil {
		t.Fatalf("Open #2: %v", err)
	}
	c.now = func() time.Time { return time.Unix(1e9+1, 0) }
	c.Get(id)

	id2 := ActionID(dummyID(2))
	c.Get(id2)
	c.PutBytes(id2, []byte("abc"))
	c.Get(id2)
	c.Get(id)

	data, err := ioutil.ReadFile(filepath.Join(dir, "log.txt"))
	if err != nil {
		t.Fatal(err)
	}
	want := `1000000000 miss 0100000000000000000000000000000000000000000000000000000000000000
1000000000 put 0100000000000000000000000000000000000000000000000000000000000000 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad 3
1000000000 get 0100000000000000000000000000000000000000000000000000000000000000
1000000001 get 0100000000000000000000000000000000000000000000000000000000000000
1000000001 miss 0200000000000000000000000000000000000000000000000000000000000000
1000000001 put 0200000000000000000000000000000000000000000000000000000000000000 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad 3
1000000001 get 0200000000000000000000000000000000000000000000000000000000000000
1000000001 get 0100000000000000000000000000000000000000000000000000000000000000
`
	if string(data) != want {
		t.Fatalf("log:\n%s\nwant:\n%s", string(data), want)
	}
}

func dummyID(x int) [HashSize]byte {
	var out [HashSize]byte
	binary.LittleEndian.PutUint64(out[:], uint64(x))
	return out
}
