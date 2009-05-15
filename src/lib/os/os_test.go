// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"fmt";
	"io";
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

func size(name string, t *testing.T) uint64 {
	file, err := Open(name, O_RDONLY, 0);
	defer file.Close();
	if err != nil {
		t.Fatal("open failed:", err);
	}
	var buf [100]byte;
	len := 0;
	for {
		n, e := file.Read(&buf);
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
	file, err1 := Open("/etc/passwd", O_RDONLY, 0);
	defer file.Close();
	if err1 != nil {
		t.Fatal("open failed:", err1);
	}
	dir, err2 := file.Stat();
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
	file, err := Open(dir, O_RDONLY, 0);
	defer file.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err);
	}
	s, err2 := file.Readdirnames(-1);
	if err2 != nil {
		t.Fatalf("readdirnames %q failed: %v", err2);
	}
	for i, m := range contents {
		found := false;
		for j, n := range s {
			if n == "." || n == ".." {
				t.Errorf("got %s in directory", n);
			}
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
	file, err := Open(dir, O_RDONLY, 0);
	defer file.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err);
	}
	s, err2 := file.Readdir(-1);
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
func smallReaddirnames(file *File, length int, t *testing.T) []string {
	names := make([]string, length);
	count := 0;
	for {
		d, err := file.Readdirnames(1);
		if err != nil {
			t.Fatalf("readdir %q failed: %v", file.Name(), err);
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
	file, err := Open(dir, O_RDONLY, 0);
	defer file.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err);
	}
	all, err1 := file.Readdirnames(-1);
	if err1 != nil {
		t.Fatalf("readdirnames %q failed: %v", dir, err1);
	}
	file1, err2 := Open(dir, O_RDONLY, 0);
	if err2 != nil {
		t.Fatalf("open %q failed: %v", dir, err2);
	}
	small := smallReaddirnames(file1, len(all)+100, t);	// +100 in case we screw up
	for i, n := range all {
		if small[i] != n {
			t.Errorf("small read %q %q mismatch: %v", small[i], n);
		}
	}
}

func TestHardLink(t *testing.T) {
	from, to := "hardlinktestfrom", "hardlinktestto";
	Remove(from); // Just in case.
	file, err := Open(to, O_CREAT | O_WRONLY, 0666);
	if err != nil {
		t.Fatalf("open %q failed: %v", to, err);
	}
	defer Remove(to);
	if err = file.Close(); err != nil {
		t.Errorf("close %q failed: %v", to, err);
	}
	err = Link(to, from);
	if err != nil {
		t.Fatalf("link %q, %q failed: %v", to, from, err);
	}
	defer Remove(from);
	tostat, err := Stat(to);
	if err != nil {
		t.Fatalf("stat %q failed: %v", to, err);
	}
	fromstat, err := Stat(from);
	if err != nil {
		t.Fatalf("stat %q failed: %v", from, err);
	}
	if tostat.Dev != fromstat.Dev || tostat.Ino != fromstat.Ino {
		t.Errorf("link %q, %q did not create hard link", to, from);
	}
}

func TestSymLink(t *testing.T) {
	from, to := "symlinktestfrom", "symlinktestto";
	Remove(from); // Just in case.
	file, err := Open(to, O_CREAT | O_WRONLY, 0666);
	if err != nil {
		t.Fatalf("open %q failed: %v", to, err);
	}
	defer Remove(to);
	if err = file.Close(); err != nil {
		t.Errorf("close %q failed: %v", to, err);
	}
	err = Symlink(to, from);
	if err != nil {
		t.Fatalf("symlink %q, %q failed: %v", to, from, err);
	}
	defer Remove(from);
	tostat, err := Stat(to);
	if err != nil {
		t.Fatalf("stat %q failed: %v", to, err);
	}
	if tostat.FollowedSymlink {
		t.Fatalf("stat %q claims to have followed a symlink", to);
	}
	fromstat, err := Stat(from);
	if err != nil {
		t.Fatalf("stat %q failed: %v", from, err);
	}
	if tostat.Dev != fromstat.Dev || tostat.Ino != fromstat.Ino {
		t.Errorf("symlink %q, %q did not create symlink", to, from);
	}
	fromstat, err = Lstat(from);
	if err != nil {
		t.Fatalf("lstat %q failed: %v", from, err);
	}
	if !fromstat.IsSymlink() {
		t.Fatalf("symlink %q, %q did not create symlink", to, from);
	}
	fromstat, err = Stat(from);
	if err != nil {
		t.Fatalf("stat %q failed: %v", from, err);
	}
	if !fromstat.FollowedSymlink {
		t.Fatalf("stat %q did not follow symlink");
	}
	s, err := Readlink(from);
	if err != nil {
		t.Fatalf("readlink %q failed: %v", from, err);
	}
	if s != to {
		t.Fatalf("after symlink %q != %q", s, to);
	}
	file, err = Open(from, O_RDONLY, 0);
	if err != nil {
		t.Fatalf("open %q failed: %v", from, err);
	}
	file.Close();
}

func TestLongSymlink(t *testing.T) {
	s := "0123456789abcdef";
	s = s + s + s + s + s + s + s + s + s + s + s + s + s + s + s + s + s;
	from := "longsymlinktestfrom";
	err := Symlink(s, from);
	if err != nil {
		t.Fatalf("symlink %q, %q failed: %v", s, from, err);
	}
	defer Remove(from);
	r, err := Readlink(from);
	if err != nil {
		t.Fatalf("readlink %q failed: %v", from, err);
	}
	if r != s {
		t.Fatalf("after symlink %q != %q", r, s);
	}
}

func TestForkExec(t *testing.T) {
	r, w, err := Pipe();
	if err != nil {
		t.Fatalf("Pipe: %v", err);
	}
	pid, err := ForkExec("/bin/pwd", []string{"pwd"}, nil, "/", []*File{nil, w, os.Stderr});
	if err != nil {
		t.Fatalf("ForkExec: %v", err);
	}
	w.Close();

	var b io.ByteBuffer;
	io.Copy(r, &b);
	output := string(b.Data());
	expect := "/\n";
	if output != expect {
		t.Errorf("exec /bin/pwd returned %q wanted %q", output, expect);
	}
	Wait(pid, 0);
}

func checkMode(t *testing.T, path string, mode uint32) {
	dir, err := Stat(path);
	if err != nil {
		t.Fatalf("Stat %q (looking for mode %#o): %s", path, mode, err);
	}
	if dir.Mode & 0777 != mode {
		t.Errorf("Stat %q: mode %#o want %#o", path, dir.Mode, 0777);
	}
}

func TestChmod(t *testing.T) {
	MkdirAll("_obj", 0777);
	const Path = "_obj/_TestChmod_";
	fd, err := os.Open(Path, os.O_WRONLY | os.O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %s: %s", Path, err);
	}

	if err = os.Chmod(Path, 0456); err != nil {
		t.Fatalf("chmod %s 0456: %s", Path, err);
	}
	checkMode(t, Path, 0456);

	if err = fd.Chmod(0123); err != nil {
		t.Fatalf("fchmod %s 0123: %s", Path, err);
	}
	checkMode(t, Path, 0123);

	fd.Close();
	Remove(Path);
}

func checkUidGid(t *testing.T, path string, uid, gid int) {
	dir, err := Stat(path);
	if err != nil {
		t.Fatalf("Stat %q (looking for uid/gid %#o/%#o): %s", path, uid, gid, err);
	}
	if dir.Uid != uint32(uid) {
		t.Errorf("Stat %q: uid %#o want %#o", path, dir.Uid, uid);
	}
	if dir.Gid != uint32(gid) {
		t.Errorf("Stat %q: gid %#o want %#o", path, dir.Gid, uid);
	}
}

func TestChown(t *testing.T) {
	// Use /tmp, not _obj, to make sure we're on a local file system,
	// so that the group ids returned by Getgroups will be allowed
	// on the file.  If _obj is on NFS, the Getgroups groups are
	// basically useless.

	const Path = "/tmp/_TestChown_";
	fd, err := os.Open(Path, os.O_WRONLY | os.O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %s: %s", Path, err);
	}
	dir, err := fd.Stat();
	if err != nil {
		t.Fatalf("fstat %s: %s", Path, err);
	}
	defer fd.Close();
	defer Remove(Path);

	// Can't change uid unless root, but can try
	// changing the group id.  First try our current group.
	gid := Getgid();
	if err = os.Chown(Path, -1, gid); err != nil {
		t.Fatalf("chown %s -1 %d: %s", Path, gid, err);
	}
	checkUidGid(t, Path, int(dir.Uid), gid);

	// Then try all the auxiliary groups.
	groups, err := Getgroups();
	if err != nil {
		t.Fatalf("getgroups: %s", err);
	}
	for i, g := range groups {
		if err = os.Chown(Path, -1, g); err != nil {
			t.Fatalf("chown %s -1 %d: %s", Path, g, err);
		}
		checkUidGid(t, Path, int(dir.Uid), g);

		// change back to gid to test fd.Chown
		if err = fd.Chown(-1, gid); err != nil {
			t.Fatalf("fchown %s -1 %d: %s", Path, gid, err);
		}
		checkUidGid(t, Path, int(dir.Uid), gid);
	}
}

func checkSize(t *testing.T, path string, size uint64) {
	dir, err := Stat(path);
	if err != nil {
		t.Fatalf("Stat %q (looking for size %d): %s", path, size, err);
	}
	if dir.Size != size {
		t.Errorf("Stat %q: size %d want %d", path, dir.Size, size);
	}
}

func TestTruncate(t *testing.T) {
	MkdirAll("_obj", 0777);
	const Path = "_obj/_TestTruncate_";
	fd, err := os.Open(Path, os.O_WRONLY | os.O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %s: %s", Path, err);
	}

	checkSize(t, Path, 0);
	fd.Write(io.StringBytes("hello, world\n"));
	checkSize(t, Path, 13);
	fd.Truncate(10);
	checkSize(t, Path, 10);
	fd.Truncate(1024);
	checkSize(t, Path, 1024);
	fd.Truncate(0);
	checkSize(t, Path, 0);
	fd.Write(io.StringBytes("surprise!"));
	checkSize(t, Path, 13 + 9);	// wrote at offset past where hello, world was.
	fd.Close();
	Remove(Path);
}
