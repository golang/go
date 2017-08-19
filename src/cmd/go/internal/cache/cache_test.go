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
)

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
	if err := c1.putIndexEntry(dummyID(1), dummyID(12), 13); err != nil {
		t.Fatalf("addIndexEntry: %v", err)
	}
	if err := c1.putIndexEntry(dummyID(1), dummyID(2), 3); err != nil { // overwrite entry
		t.Fatalf("addIndexEntry: %v", err)
	}
	if out, size, err := c1.Get(dummyID(1)); err != nil || out != dummyID(2) || size != 3 {
		t.Fatalf("c1.Get(1) = %x, %v, %v, want %x, %v, nil", out[:], size, err, dummyID(2), 3)
	}

	c2, err := Open(cdir)
	if err != nil {
		t.Fatalf("Open(c2) (reuse): %v", err)
	}
	if out, size, err := c2.Get(dummyID(1)); err != nil || out != dummyID(2) || size != 3 {
		t.Fatalf("c2.Get(1) = %x, %v, %v, want %x, %v, nil", out[:], size, err, dummyID(2), 3)
	}
	if err := c2.putIndexEntry(dummyID(2), dummyID(3), 4); err != nil {
		t.Fatalf("addIndexEntry: %v", err)
	}
	if out, size, err := c1.Get(dummyID(2)); err != nil || out != dummyID(3) || size != 4 {
		t.Fatalf("c1.Get(2) = %x, %v, %v, want %x, %v, nil", out[:], size, err, dummyID(3), 4)
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
		if err := c.putIndexEntry(dummyID(i), dummyID(i*99), int64(i)*101); err != nil {
			t.Fatalf("addIndexEntry: %v", err)
		}
		id := ActionID(dummyID(i))
		out, size, err := c.Get(id)
		if err != nil {
			t.Fatalf("Get(%x): %v", id, err)
		}
		if out != dummyID(i*99) || size != int64(i)*101 {
			t.Errorf("Get(%x) = %x, %d, want %x, %d", id, out, size, dummyID(i*99), int64(i)*101)
		}
	}
	for i := 0; i < n; i++ {
		id := ActionID(dummyID(i))
		out, size, err := c.Get(id)
		if err != nil {
			t.Fatalf("Get2(%x): %v", id, err)
		}
		if out != dummyID(i*99) || size != int64(i)*101 {
			t.Errorf("Get2(%x) = %x, %d, want %x, %d", id, out, size, dummyID(i*99), int64(i)*101)
		}
	}
}

func dummyID(x int) [HashSize]byte {
	var out [HashSize]byte
	binary.LittleEndian.PutUint64(out[:], uint64(x))
	return out
}
