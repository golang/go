// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"fmt";
	"os";
	"testing";
)

var dot = []string{
	"dir_amd64_darwin.go",
	"dir_amd64_linux.go",
	"env.go",
	"error.go",
	"file.go",
	"os_test.go",
	"time.go",
	"types.go",
	"stat_amd64_darwin.go",
	"stat_amd64_linux.go"
}

var etc = []string{
	"group",
	"hosts",
	"passwd",
}

func size(file string, t *testing.T) uint64 {
	fd, err := Open(file, O_RDONLY, 0);
	defer fd.Close();
	if err != nil {
		t.Fatal("open failed:", err);
	}
	var buf [100]byte;
	len := 0;
	for {
		n, e := fd.Read(buf);
		if n < 0 || e != nil {
			t.Fatal("read failed:", err);
		}
		if n == 0 {
			break
		}
		len += n;
	}
	return uint64(len)
}

func TestStat(t *testing.T) {
	dir, err := Stat("/etc/passwd");
	if err != nil {
		t.Fatal("stat failed:", err);
	}
	if dir.Name != "passwd" {
		t.Error("name should be passwd; is", dir.Name);
	}
	filesize := size("/etc/passwd", t);
	if dir.Size != filesize {
		t.Error("size should be ", filesize, "; is", dir.Size);
	}
}

func TestFstat(t *testing.T) {
	fd, err1 := Open("/etc/passwd", O_RDONLY, 0);
	defer fd.Close();
	if err1 != nil {
		t.Fatal("open failed:", err1);
	}
	dir, err2 := Fstat(fd);
	if err2 != nil {
		t.Fatal("fstat failed:", err2);
	}
	if dir.Name != "passwd" {
		t.Error("name should be passwd; is", dir.Name);
	}
	filesize := size("/etc/passwd", t);
	if dir.Size != filesize {
		t.Error("size should be ", filesize, "; is", dir.Size);
	}
}

func TestLstat(t *testing.T) {
	dir, err := Lstat("/etc/passwd");
	if err != nil {
		t.Fatal("lstat failed:", err);
	}
	if dir.Name != "passwd" {
		t.Error("name should be passwd; is", dir.Name);
	}
	filesize := size("/etc/passwd", t);
	if dir.Size != filesize {
		t.Error("size should be ", filesize, "; is", dir.Size);
	}
}

func testReaddirnames(dir string, contents []string, t *testing.T) {
	fd, err := Open(dir, O_RDONLY, 0);
	defer fd.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err);
	}
	s, err2 := Readdirnames(fd, -1);
	if err2 != nil {
		t.Fatalf("readdirnames %q failed: %v", err2);
	}
	for i, m := range contents {
		found := false;
		for j, n := range s {
			if m == n {
				if found {
					t.Error("present twice:", m);
				}
				found = true
			}
		}
		if !found {
			t.Error("could not find", m);
		}
	}
}

func testReaddir(dir string, contents []string, t *testing.T) {
	fd, err := Open(dir, O_RDONLY, 0);
	defer fd.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err);
	}
	s, err2 := Readdir(fd, -1);
	if err2 != nil {
		t.Fatalf("readdir %q failed: %v", dir, err2);
	}
	for i, m := range contents {
		found := false;
		for j, n := range s {
			if m == n.Name {
				if found {
					t.Error("present twice:", m);
				}
				found = true
			}
		}
		if !found {
			t.Error("could not find", m);
		}
	}
}

func TestReaddirnames(t *testing.T) {
	testReaddirnames(".", dot, t);
	testReaddirnames("/etc", etc, t);
}

func TestReaddir(t *testing.T) {
	testReaddir(".", dot, t);
	testReaddir("/etc", etc, t);
}

// Read the directory one entry at a time.
func smallReaddirnames(fd *FD, length int, t *testing.T) []string {
	names := make([]string, length);
	count := 0;
	for {
		d, err := Readdirnames(fd, 1);
		if err != nil {
			t.Fatalf("readdir %q failed: %v", fd.Name(), err);
		}
		if len(d) == 0 {
			break
		}
		names[count] = d[0];
		count++;
	}
	return names[0:count]
}

// Check that reading a directory one entry at a time gives the same result
// as reading it all at once.
func TestReaddirnamesOneAtATime(t *testing.T) {
	dir := "/usr/bin";	// big directory that doesn't change often.
	fd, err := Open(dir, O_RDONLY, 0);
	defer fd.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err);
	}
	all, err1 := Readdirnames(fd, -1);
	if err1 != nil {
		t.Fatalf("readdirnames %q failed: %v", dir, err1);
	}
	fd1, err2 := Open(dir, O_RDONLY, 0);
	if err2 != nil {
		t.Fatalf("open %q failed: %v", dir, err2);
	}
	small := smallReaddirnames(fd1, len(all)+100, t);	// +100 in case we screw up
	for i, n := range all {
		if small[i] != n {
			t.Errorf("small read %q %q mismatch: %v", small[i], n);
		}
	}
}

