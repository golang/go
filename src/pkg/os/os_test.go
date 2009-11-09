// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes";
	"fmt";
	"io";
	. "os";
	"strings";
	"testing";
)

var dot = []string{
	"dir_darwin.go",
	"dir_linux.go",
	"env.go",
	"error.go",
	"file.go",
	"os_test.go",
	"time.go",
	"types.go",
	"stat_darwin.go",
	"stat_linux.go",
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
		t.Fatal("open failed:", err)
	}
	var buf [100]byte;
	len := 0;
	for {
		n, e := file.Read(&buf);
		len += n;
		if e == EOF {
			break
		}
		if e != nil {
			t.Fatal("read failed:", err)
		}
	}
	return uint64(len);
}

func TestStat(t *testing.T) {
	dir, err := Stat("/etc/passwd");
	if err != nil {
		t.Fatal("stat failed:", err)
	}
	if dir.Name != "passwd" {
		t.Error("name should be passwd; is", dir.Name)
	}
	filesize := size("/etc/passwd", t);
	if dir.Size != filesize {
		t.Error("size should be", filesize, "; is", dir.Size)
	}
}

func TestFstat(t *testing.T) {
	file, err1 := Open("/etc/passwd", O_RDONLY, 0);
	defer file.Close();
	if err1 != nil {
		t.Fatal("open failed:", err1)
	}
	dir, err2 := file.Stat();
	if err2 != nil {
		t.Fatal("fstat failed:", err2)
	}
	if dir.Name != "passwd" {
		t.Error("name should be passwd; is", dir.Name)
	}
	filesize := size("/etc/passwd", t);
	if dir.Size != filesize {
		t.Error("size should be", filesize, "; is", dir.Size)
	}
}

func TestLstat(t *testing.T) {
	dir, err := Lstat("/etc/passwd");
	if err != nil {
		t.Fatal("lstat failed:", err)
	}
	if dir.Name != "passwd" {
		t.Error("name should be passwd; is", dir.Name)
	}
	filesize := size("/etc/passwd", t);
	if dir.Size != filesize {
		t.Error("size should be", filesize, "; is", dir.Size)
	}
}

func testReaddirnames(dir string, contents []string, t *testing.T) {
	file, err := Open(dir, O_RDONLY, 0);
	defer file.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err)
	}
	s, err2 := file.Readdirnames(-1);
	if err2 != nil {
		t.Fatalf("readdirnames %q failed: %v", err2)
	}
	for _, m := range contents {
		found := false;
		for _, n := range s {
			if n == "." || n == ".." {
				t.Errorf("got %s in directory", n)
			}
			if m == n {
				if found {
					t.Error("present twice:", m)
				}
				found = true;
			}
		}
		if !found {
			t.Error("could not find", m)
		}
	}
}

func testReaddir(dir string, contents []string, t *testing.T) {
	file, err := Open(dir, O_RDONLY, 0);
	defer file.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err)
	}
	s, err2 := file.Readdir(-1);
	if err2 != nil {
		t.Fatalf("readdir %q failed: %v", dir, err2)
	}
	for _, m := range contents {
		found := false;
		for _, n := range s {
			if m == n.Name {
				if found {
					t.Error("present twice:", m)
				}
				found = true;
			}
		}
		if !found {
			t.Error("could not find", m)
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
			t.Fatalf("readdir %q failed: %v", file.Name(), err)
		}
		if len(d) == 0 {
			break
		}
		names[count] = d[0];
		count++;
	}
	return names[0:count];
}

// Check that reading a directory one entry at a time gives the same result
// as reading it all at once.
func TestReaddirnamesOneAtATime(t *testing.T) {
	dir := "/usr/bin";	// big directory that doesn't change often.
	file, err := Open(dir, O_RDONLY, 0);
	defer file.Close();
	if err != nil {
		t.Fatalf("open %q failed: %v", dir, err)
	}
	all, err1 := file.Readdirnames(-1);
	if err1 != nil {
		t.Fatalf("readdirnames %q failed: %v", dir, err1)
	}
	file1, err2 := Open(dir, O_RDONLY, 0);
	if err2 != nil {
		t.Fatalf("open %q failed: %v", dir, err2)
	}
	small := smallReaddirnames(file1, len(all)+100, t);	// +100 in case we screw up
	for i, n := range all {
		if small[i] != n {
			t.Errorf("small read %q %q mismatch: %v", small[i], n)
		}
	}
}

func TestHardLink(t *testing.T) {
	from, to := "hardlinktestfrom", "hardlinktestto";
	Remove(from);	// Just in case.
	file, err := Open(to, O_CREAT|O_WRONLY, 0666);
	if err != nil {
		t.Fatalf("open %q failed: %v", to, err)
	}
	defer Remove(to);
	if err = file.Close(); err != nil {
		t.Errorf("close %q failed: %v", to, err)
	}
	err = Link(to, from);
	if err != nil {
		t.Fatalf("link %q, %q failed: %v", to, from, err)
	}
	defer Remove(from);
	tostat, err := Stat(to);
	if err != nil {
		t.Fatalf("stat %q failed: %v", to, err)
	}
	fromstat, err := Stat(from);
	if err != nil {
		t.Fatalf("stat %q failed: %v", from, err)
	}
	if tostat.Dev != fromstat.Dev || tostat.Ino != fromstat.Ino {
		t.Errorf("link %q, %q did not create hard link", to, from)
	}
}

func TestSymLink(t *testing.T) {
	from, to := "symlinktestfrom", "symlinktestto";
	Remove(from);	// Just in case.
	file, err := Open(to, O_CREAT|O_WRONLY, 0666);
	if err != nil {
		t.Fatalf("open %q failed: %v", to, err)
	}
	defer Remove(to);
	if err = file.Close(); err != nil {
		t.Errorf("close %q failed: %v", to, err)
	}
	err = Symlink(to, from);
	if err != nil {
		t.Fatalf("symlink %q, %q failed: %v", to, from, err)
	}
	defer Remove(from);
	tostat, err := Stat(to);
	if err != nil {
		t.Fatalf("stat %q failed: %v", to, err)
	}
	if tostat.FollowedSymlink {
		t.Fatalf("stat %q claims to have followed a symlink", to)
	}
	fromstat, err := Stat(from);
	if err != nil {
		t.Fatalf("stat %q failed: %v", from, err)
	}
	if tostat.Dev != fromstat.Dev || tostat.Ino != fromstat.Ino {
		t.Errorf("symlink %q, %q did not create symlink", to, from)
	}
	fromstat, err = Lstat(from);
	if err != nil {
		t.Fatalf("lstat %q failed: %v", from, err)
	}
	if !fromstat.IsSymlink() {
		t.Fatalf("symlink %q, %q did not create symlink", to, from)
	}
	fromstat, err = Stat(from);
	if err != nil {
		t.Fatalf("stat %q failed: %v", from, err)
	}
	if !fromstat.FollowedSymlink {
		t.Fatalf("stat %q did not follow symlink")
	}
	s, err := Readlink(from);
	if err != nil {
		t.Fatalf("readlink %q failed: %v", from, err)
	}
	if s != to {
		t.Fatalf("after symlink %q != %q", s, to)
	}
	file, err = Open(from, O_RDONLY, 0);
	if err != nil {
		t.Fatalf("open %q failed: %v", from, err)
	}
	file.Close();
}

func TestLongSymlink(t *testing.T) {
	s := "0123456789abcdef";
	s = s+s+s+s+s+s+s+s+s+s+s+s+s+s+s+s+s;
	from := "longsymlinktestfrom";
	err := Symlink(s, from);
	if err != nil {
		t.Fatalf("symlink %q, %q failed: %v", s, from, err)
	}
	defer Remove(from);
	r, err := Readlink(from);
	if err != nil {
		t.Fatalf("readlink %q failed: %v", from, err)
	}
	if r != s {
		t.Fatalf("after symlink %q != %q", r, s)
	}
}

func TestForkExec(t *testing.T) {
	r, w, err := Pipe();
	if err != nil {
		t.Fatalf("Pipe: %v", err)
	}
	pid, err := ForkExec("/bin/pwd", []string{"pwd"}, nil, "/", []*File{nil, w, Stderr});
	if err != nil {
		t.Fatalf("ForkExec: %v", err)
	}
	w.Close();

	var b bytes.Buffer;
	io.Copy(&b, r);
	output := b.String();
	expect := "/\n";
	if output != expect {
		t.Errorf("exec /bin/pwd returned %q wanted %q", output, expect)
	}
	Wait(pid, 0);
}

func checkMode(t *testing.T, path string, mode uint32) {
	dir, err := Stat(path);
	if err != nil {
		t.Fatalf("Stat %q (looking for mode %#o): %s", path, mode, err)
	}
	if dir.Mode & 0777 != mode {
		t.Errorf("Stat %q: mode %#o want %#o", path, dir.Mode, 0777)
	}
}

func TestChmod(t *testing.T) {
	MkdirAll("_obj", 0777);
	const Path = "_obj/_TestChmod_";
	fd, err := Open(Path, O_WRONLY|O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %s: %s", Path, err)
	}

	if err = Chmod(Path, 0456); err != nil {
		t.Fatalf("chmod %s 0456: %s", Path, err)
	}
	checkMode(t, Path, 0456);

	if err = fd.Chmod(0123); err != nil {
		t.Fatalf("fchmod %s 0123: %s", Path, err)
	}
	checkMode(t, Path, 0123);

	fd.Close();
	Remove(Path);
}

func checkUidGid(t *testing.T, path string, uid, gid int) {
	dir, err := Stat(path);
	if err != nil {
		t.Fatalf("Stat %q (looking for uid/gid %d/%d): %s", path, uid, gid, err)
	}
	if dir.Uid != uint32(uid) {
		t.Errorf("Stat %q: uid %d want %d", path, dir.Uid, uid)
	}
	if dir.Gid != uint32(gid) {
		t.Errorf("Stat %q: gid %d want %d", path, dir.Gid, gid)
	}
}

func TestChown(t *testing.T) {
	// Use /tmp, not _obj, to make sure we're on a local file system,
	// so that the group ids returned by Getgroups will be allowed
	// on the file.  If _obj is on NFS, the Getgroups groups are
	// basically useless.

	const Path = "/tmp/_TestChown_";
	fd, err := Open(Path, O_WRONLY|O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %s: %s", Path, err)
	}
	dir, err := fd.Stat();
	if err != nil {
		t.Fatalf("fstat %s: %s", Path, err)
	}
	defer fd.Close();
	defer Remove(Path);

	// Can't change uid unless root, but can try
	// changing the group id.  First try our current group.
	gid := Getgid();
	t.Log("gid:", gid);
	if err = Chown(Path, -1, gid); err != nil {
		t.Fatalf("chown %s -1 %d: %s", Path, gid, err)
	}
	checkUidGid(t, Path, int(dir.Uid), gid);

	// Then try all the auxiliary groups.
	groups, err := Getgroups();
	if err != nil {
		t.Fatalf("getgroups: %s", err)
	}
	t.Log("groups: ", groups);
	for _, g := range groups {
		if err = Chown(Path, -1, g); err != nil {
			t.Fatalf("chown %s -1 %d: %s", Path, g, err)
		}
		checkUidGid(t, Path, int(dir.Uid), g);

		// change back to gid to test fd.Chown
		if err = fd.Chown(-1, gid); err != nil {
			t.Fatalf("fchown %s -1 %d: %s", Path, gid, err)
		}
		checkUidGid(t, Path, int(dir.Uid), gid);
	}
}

func checkSize(t *testing.T, path string, size uint64) {
	dir, err := Stat(path);
	if err != nil {
		t.Fatalf("Stat %q (looking for size %d): %s", path, size, err)
	}
	if dir.Size != size {
		t.Errorf("Stat %q: size %d want %d", path, dir.Size, size)
	}
}

func TestTruncate(t *testing.T) {
	MkdirAll("_obj", 0777);
	const Path = "_obj/_TestTruncate_";
	fd, err := Open(Path, O_WRONLY|O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %s: %s", Path, err)
	}

	checkSize(t, Path, 0);
	fd.Write(strings.Bytes("hello, world\n"));
	checkSize(t, Path, 13);
	fd.Truncate(10);
	checkSize(t, Path, 10);
	fd.Truncate(1024);
	checkSize(t, Path, 1024);
	fd.Truncate(0);
	checkSize(t, Path, 0);
	fd.Write(strings.Bytes("surprise!"));
	checkSize(t, Path, 13+9);	// wrote at offset past where hello, world was.
	fd.Close();
	Remove(Path);
}

func TestChdirAndGetwd(t *testing.T) {
	fd, err := Open(".", O_RDONLY, 0);
	if err != nil {
		t.Fatalf("Open .: %s", err)
	}
	// These are chosen carefully not to be symlinks on a Mac
	// (unlike, say, /var, /etc, and /tmp).
	dirs := []string{"/bin", "/", "/usr/bin"};
	for mode := 0; mode < 2; mode++ {
		for _, d := range dirs {
			if mode == 0 {
				err = Chdir(d)
			} else {
				fd1, err := Open(d, O_RDONLY, 0);
				if err != nil {
					t.Errorf("Open %s: %s", d, err);
					continue;
				}
				err = fd1.Chdir();
				fd1.Close();
			}
			pwd, err1 := Getwd();
			err2 := fd.Chdir();
			if err2 != nil {
				// We changed the current directory and cannot go back.
				// Don't let the tests continue; they'll scribble
				// all over some other directory.
				fmt.Fprintf(Stderr, "fchdir back to dot failed: %s\n", err2);
				Exit(1);
			}
			if err != nil {
				fd.Close();
				t.Fatalf("Chdir %s: %s", d, err);
			}
			if err1 != nil {
				fd.Close();
				t.Fatalf("Getwd in %s: %s", d, err1);
			}
			if pwd != d {
				fd.Close();
				t.Fatalf("Getwd returned %q want %q", pwd, d);
			}
		}
	}
	fd.Close();
}

func TestTime(t *testing.T) {
	// Just want to check that Time() is getting something.
	// A common failure mode on Darwin is to get 0, 0,
	// because it returns the time in registers instead of
	// filling in the structure passed to the system call.
	// Too bad the compiler doesn't know that
	// 365.24*86400 is an integer.
	sec, nsec, err := Time();
	if sec < (2009-1970)*36524*864 {
		t.Errorf("Time() = %d, %d, %s; not plausible", sec, nsec, err)
	}
}

func TestSeek(t *testing.T) {
	f, err := Open("_obj/seektest", O_CREAT|O_RDWR|O_TRUNC, 0666);
	if err != nil {
		t.Fatalf("open _obj/seektest: %s", err)
	}

	const data = "hello, world\n";
	io.WriteString(f, data);

	type test struct {
		in	int64;
		whence	int;
		out	int64;
	}
	var tests = []test{
		test{0, 1, int64(len(data))},
		test{0, 0, 0},
		test{5, 0, 5},
		test{0, 2, int64(len(data))},
		test{0, 0, 0},
		test{-1, 2, int64(len(data))-1},
		test{1<<40, 0, 1<<40},
		test{1<<40, 2, 1<<40 + int64(len(data))},
	};
	for i, tt := range tests {
		off, err := f.Seek(tt.in, tt.whence);
		if off != tt.out || err != nil {
			t.Errorf("#%d: Seek(%v, %v) = %v, %v want %v, nil", i, tt.in, tt.whence, off, err, tt.out)
		}
	}
	f.Close();
}

type openErrorTest struct {
	path	string;
	mode	int;
	error	string;
}

var openErrorTests = []openErrorTest{
	openErrorTest{
		"/etc/no-such-file",
		O_RDONLY,
		"open /etc/no-such-file: no such file or directory",
	},
	openErrorTest{
		"/etc",
		O_WRONLY,
		"open /etc: is a directory",
	},
	openErrorTest{
		"/etc/passwd/group",
		O_WRONLY,
		"open /etc/passwd/group: not a directory",
	},
}

func TestOpenError(t *testing.T) {
	for _, tt := range openErrorTests {
		f, err := Open(tt.path, tt.mode, 0);
		if err == nil {
			t.Errorf("Open(%q, %d) succeeded", tt.path, tt.mode);
			f.Close();
			continue;
		}
		if s := err.String(); s != tt.error {
			t.Errorf("Open(%q, %d) = _, %q; want %q", tt.path, tt.mode, s, tt.error)
		}
	}
}

func run(t *testing.T, cmd []string) string {
	// Run /bin/hostname and collect output.
	r, w, err := Pipe();
	if err != nil {
		t.Fatal(err)
	}
	pid, err := ForkExec("/bin/hostname", []string{"hostname"}, nil, "/", []*File{nil, w, Stderr});
	if err != nil {
		t.Fatal(err)
	}
	w.Close();

	var b bytes.Buffer;
	io.Copy(&b, r);
	Wait(pid, 0);
	output := b.String();
	if n := len(output); n > 0 && output[n-1] == '\n' {
		output = output[0 : n-1]
	}
	if output == "" {
		t.Fatalf("%v produced no output", cmd)
	}

	return output;
}


func TestHostname(t *testing.T) {
	// Check internal Hostname() against the output of /bin/hostname.
	hostname, err := Hostname();
	if err != nil {
		t.Fatalf("%v", err)
	}
	want := run(t, []string{"/bin/hostname"});
	if hostname != want {
		t.Errorf("Hostname() = %q, want %q", hostname, want)
	}
}

func TestReadAt(t *testing.T) {
	f, err := Open("_obj/readtest", O_CREAT|O_RDWR|O_TRUNC, 0666);
	if err != nil {
		t.Fatalf("open _obj/readtest: %s", err)
	}
	const data = "hello, world\n";
	io.WriteString(f, data);

	b := make([]byte, 5);
	n, err := f.ReadAt(b, 7);
	if err != nil || n != len(b) {
		t.Fatalf("ReadAt 7: %d, %r", n, err)
	}
	if string(b) != "world" {
		t.Fatalf("ReadAt 7: have %q want %q", string(b), "world")
	}
}

func TestWriteAt(t *testing.T) {
	f, err := Open("_obj/writetest", O_CREAT|O_RDWR|O_TRUNC, 0666);
	if err != nil {
		t.Fatalf("open _obj/writetest: %s", err)
	}
	const data = "hello, world\n";
	io.WriteString(f, data);

	n, err := f.WriteAt(strings.Bytes("WORLD"), 7);
	if err != nil || n != 5 {
		t.Fatalf("WriteAt 7: %d, %v", n, err)
	}

	b, err := io.ReadFile("_obj/writetest");
	if err != nil {
		t.Fatalf("ReadFile _obj/writetest: %v", err)
	}
	if string(b) != "hello, WORLD\n" {
		t.Fatalf("after write: have %q want %q", string(b), "hello, WORLD\n")
	}
}
