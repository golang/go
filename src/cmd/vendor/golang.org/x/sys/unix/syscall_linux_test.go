// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package unix_test

import (
	"os"
	"runtime"
	"runtime/debug"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestIoctlGetInt(t *testing.T) {
	f, err := os.Open("/dev/random")
	if err != nil {
		t.Fatalf("failed to open device: %v", err)
	}
	defer f.Close()

	v, err := unix.IoctlGetInt(int(f.Fd()), unix.RNDGETENTCNT)
	if err != nil {
		t.Fatalf("failed to perform ioctl: %v", err)
	}

	t.Logf("%d bits of entropy available", v)
}

func TestPpoll(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("mkfifo syscall is not available on android, skipping test")
	}

	f, cleanup := mktmpfifo(t)
	defer cleanup()

	const timeout = 100 * time.Millisecond

	ok := make(chan bool, 1)
	go func() {
		select {
		case <-time.After(10 * timeout):
			t.Errorf("Ppoll: failed to timeout after %d", 10*timeout)
		case <-ok:
		}
	}()

	fds := []unix.PollFd{{Fd: int32(f.Fd()), Events: unix.POLLIN}}
	timeoutTs := unix.NsecToTimespec(int64(timeout))
	n, err := unix.Ppoll(fds, &timeoutTs, nil)
	ok <- true
	if err != nil {
		t.Errorf("Ppoll: unexpected error: %v", err)
		return
	}
	if n != 0 {
		t.Errorf("Ppoll: wrong number of events: got %v, expected %v", n, 0)
		return
	}
}

func TestTime(t *testing.T) {
	var ut unix.Time_t
	ut2, err := unix.Time(&ut)
	if err != nil {
		t.Fatalf("Time: %v", err)
	}
	if ut != ut2 {
		t.Errorf("Time: return value %v should be equal to argument %v", ut2, ut)
	}

	var now time.Time

	for i := 0; i < 10; i++ {
		ut, err = unix.Time(nil)
		if err != nil {
			t.Fatalf("Time: %v", err)
		}

		now = time.Now()

		if int64(ut) == now.Unix() {
			return
		}
	}

	t.Errorf("Time: return value %v should be nearly equal to time.Now().Unix() %v", ut, now.Unix())
}

func TestUtime(t *testing.T) {
	defer chtmpdir(t)()

	touch(t, "file1")

	buf := &unix.Utimbuf{
		Modtime: 12345,
	}

	err := unix.Utime("file1", buf)
	if err != nil {
		t.Fatalf("Utime: %v", err)
	}

	fi, err := os.Stat("file1")
	if err != nil {
		t.Fatal(err)
	}

	if fi.ModTime().Unix() != 12345 {
		t.Errorf("Utime: failed to change modtime: expected %v, got %v", 12345, fi.ModTime().Unix())
	}
}

func TestUtimesNanoAt(t *testing.T) {
	defer chtmpdir(t)()

	symlink := "symlink1"
	os.Remove(symlink)
	err := os.Symlink("nonexisting", symlink)
	if err != nil {
		t.Fatal(err)
	}

	ts := []unix.Timespec{
		{Sec: 1111, Nsec: 2222},
		{Sec: 3333, Nsec: 4444},
	}
	err = unix.UtimesNanoAt(unix.AT_FDCWD, symlink, ts, unix.AT_SYMLINK_NOFOLLOW)
	if err != nil {
		t.Fatalf("UtimesNanoAt: %v", err)
	}

	var st unix.Stat_t
	err = unix.Lstat(symlink, &st)
	if err != nil {
		t.Fatalf("Lstat: %v", err)
	}
	if st.Atim != ts[0] {
		t.Errorf("UtimesNanoAt: wrong atime: %v", st.Atim)
	}
	if st.Mtim != ts[1] {
		t.Errorf("UtimesNanoAt: wrong mtime: %v", st.Mtim)
	}
}

func TestRlimitAs(t *testing.T) {
	// disable GC during to avoid flaky test
	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	var rlim unix.Rlimit
	err := unix.Getrlimit(unix.RLIMIT_AS, &rlim)
	if err != nil {
		t.Fatalf("Getrlimit: %v", err)
	}
	var zero unix.Rlimit
	if zero == rlim {
		t.Fatalf("Getrlimit: got zero value %#v", rlim)
	}
	set := rlim
	set.Cur = uint64(unix.Getpagesize())
	err = unix.Setrlimit(unix.RLIMIT_AS, &set)
	if err != nil {
		t.Fatalf("Setrlimit: set failed: %#v %v", set, err)
	}

	// RLIMIT_AS was set to the page size, so mmap()'ing twice the page size
	// should fail. See 'man 2 getrlimit'.
	_, err = unix.Mmap(-1, 0, 2*unix.Getpagesize(), unix.PROT_NONE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err == nil {
		t.Fatal("Mmap: unexpectedly suceeded after setting RLIMIT_AS")
	}

	err = unix.Setrlimit(unix.RLIMIT_AS, &rlim)
	if err != nil {
		t.Fatalf("Setrlimit: restore failed: %#v %v", rlim, err)
	}

	b, err := unix.Mmap(-1, 0, 2*unix.Getpagesize(), unix.PROT_NONE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("Mmap: %v", err)
	}
	err = unix.Munmap(b)
	if err != nil {
		t.Fatalf("Munmap: %v", err)
	}
}

func TestSelect(t *testing.T) {
	_, err := unix.Select(0, nil, nil, nil, &unix.Timeval{Sec: 0, Usec: 0})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	dur := 150 * time.Millisecond
	tv := unix.NsecToTimeval(int64(dur))
	start := time.Now()
	_, err = unix.Select(0, nil, nil, nil, &tv)
	took := time.Since(start)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	if took < dur {
		t.Errorf("Select: timeout should have been at least %v, got %v", dur, took)
	}
}

func TestPselect(t *testing.T) {
	_, err := unix.Pselect(0, nil, nil, nil, &unix.Timespec{Sec: 0, Nsec: 0}, nil)
	if err != nil {
		t.Fatalf("Pselect: %v", err)
	}

	dur := 2500 * time.Microsecond
	ts := unix.NsecToTimespec(int64(dur))
	start := time.Now()
	_, err = unix.Pselect(0, nil, nil, nil, &ts, nil)
	took := time.Since(start)
	if err != nil {
		t.Fatalf("Pselect: %v", err)
	}

	if took < dur {
		t.Errorf("Pselect: timeout should have been at least %v, got %v", dur, took)
	}
}

func TestSchedSetaffinity(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var oldMask unix.CPUSet
	err := unix.SchedGetaffinity(0, &oldMask)
	if err != nil {
		t.Fatalf("SchedGetaffinity: %v", err)
	}

	var newMask unix.CPUSet
	newMask.Zero()
	if newMask.Count() != 0 {
		t.Errorf("CpuZero: didn't zero CPU set: %v", newMask)
	}
	cpu := 1
	newMask.Set(cpu)
	if newMask.Count() != 1 || !newMask.IsSet(cpu) {
		t.Errorf("CpuSet: didn't set CPU %d in set: %v", cpu, newMask)
	}
	cpu = 5
	newMask.Set(cpu)
	if newMask.Count() != 2 || !newMask.IsSet(cpu) {
		t.Errorf("CpuSet: didn't set CPU %d in set: %v", cpu, newMask)
	}
	newMask.Clear(cpu)
	if newMask.Count() != 1 || newMask.IsSet(cpu) {
		t.Errorf("CpuClr: didn't clear CPU %d in set: %v", cpu, newMask)
	}

	if runtime.NumCPU() < 2 {
		t.Skip("skipping setaffinity tests on single CPU system")
	}
	if runtime.GOOS == "android" {
		t.Skip("skipping setaffinity tests on android")
	}

	err = unix.SchedSetaffinity(0, &newMask)
	if err != nil {
		t.Fatalf("SchedSetaffinity: %v", err)
	}

	var gotMask unix.CPUSet
	err = unix.SchedGetaffinity(0, &gotMask)
	if err != nil {
		t.Fatalf("SchedGetaffinity: %v", err)
	}

	if gotMask != newMask {
		t.Errorf("SchedSetaffinity: returned affinity mask does not match set affinity mask")
	}

	// Restore old mask so it doesn't affect successive tests
	err = unix.SchedSetaffinity(0, &oldMask)
	if err != nil {
		t.Fatalf("SchedSetaffinity: %v", err)
	}
}

func TestStatx(t *testing.T) {
	var stx unix.Statx_t
	err := unix.Statx(unix.AT_FDCWD, ".", 0, 0, &stx)
	if err == unix.ENOSYS || err == unix.EPERM {
		t.Skip("statx syscall is not available, skipping test")
	} else if err != nil {
		t.Fatalf("Statx: %v", err)
	}

	defer chtmpdir(t)()
	touch(t, "file1")

	var st unix.Stat_t
	err = unix.Stat("file1", &st)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}

	flags := unix.AT_STATX_SYNC_AS_STAT
	err = unix.Statx(unix.AT_FDCWD, "file1", flags, unix.STATX_ALL, &stx)
	if err != nil {
		t.Fatalf("Statx: %v", err)
	}

	if uint32(stx.Mode) != st.Mode {
		t.Errorf("Statx: returned stat mode does not match Stat")
	}

	ctime := unix.StatxTimestamp{Sec: int64(st.Ctim.Sec), Nsec: uint32(st.Ctim.Nsec)}
	mtime := unix.StatxTimestamp{Sec: int64(st.Mtim.Sec), Nsec: uint32(st.Mtim.Nsec)}

	if stx.Ctime != ctime {
		t.Errorf("Statx: returned stat ctime does not match Stat")
	}
	if stx.Mtime != mtime {
		t.Errorf("Statx: returned stat mtime does not match Stat")
	}

	err = os.Symlink("file1", "symlink1")
	if err != nil {
		t.Fatal(err)
	}

	err = unix.Lstat("symlink1", &st)
	if err != nil {
		t.Fatalf("Lstat: %v", err)
	}

	err = unix.Statx(unix.AT_FDCWD, "symlink1", flags, unix.STATX_BASIC_STATS, &stx)
	if err != nil {
		t.Fatalf("Statx: %v", err)
	}

	// follow symlink, expect a regulat file
	if stx.Mode&unix.S_IFREG == 0 {
		t.Errorf("Statx: didn't follow symlink")
	}

	err = unix.Statx(unix.AT_FDCWD, "symlink1", flags|unix.AT_SYMLINK_NOFOLLOW, unix.STATX_ALL, &stx)
	if err != nil {
		t.Fatalf("Statx: %v", err)
	}

	// follow symlink, expect a symlink
	if stx.Mode&unix.S_IFLNK == 0 {
		t.Errorf("Statx: unexpectedly followed symlink")
	}
	if uint32(stx.Mode) != st.Mode {
		t.Errorf("Statx: returned stat mode does not match Lstat")
	}

	ctime = unix.StatxTimestamp{Sec: int64(st.Ctim.Sec), Nsec: uint32(st.Ctim.Nsec)}
	mtime = unix.StatxTimestamp{Sec: int64(st.Mtim.Sec), Nsec: uint32(st.Mtim.Nsec)}

	if stx.Ctime != ctime {
		t.Errorf("Statx: returned stat ctime does not match Lstat")
	}
	if stx.Mtime != mtime {
		t.Errorf("Statx: returned stat mtime does not match Lstat")
	}
}

// stringsFromByteSlice converts a sequence of attributes to a []string.
// On Linux, each entry is a NULL-terminated string.
func stringsFromByteSlice(buf []byte) []string {
	var result []string
	off := 0
	for i, b := range buf {
		if b == 0 {
			result = append(result, string(buf[off:i]))
			off = i + 1
		}
	}
	return result
}
