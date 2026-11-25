// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/lockedfile"
)

func init() {
	verify = false // even if GODEBUG is set
}

func TestBasic(t *testing.T) {
	dir := t.TempDir()
	_, err := Open(filepath.Join(dir, "notexist"))
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
	c, err := Open(t.TempDir())
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

	c, err := Open(t.TempDir())
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
	for _, trimBuildCache := range []bool{false, true} {
		t.Run(fmt.Sprintf("trimBuildCache=%v", trimBuildCache), func(t *testing.T) {
			testCacheTrim(t, trimBuildCache)
		})
	}
}

func testCacheTrim(t *testing.T, trimBuildCache bool) {
	dir := t.TempDir()
	c, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	c.trimBuildCache = trimBuildCache
	c.statfs = func(string) (avail, total, bsize uint64) { return 0, 0, 0 }
	c.sampleAcc.Store(0)
	const start = 1000000000
	now := int64(start)
	c.now = func() time.Time { return time.Unix(now, 0) }
	writeMtime := func(sec int64) int64 {
		if !trimBuildCache {
			return sec
		}
		return time.Unix(sec, 0).Truncate(time.Minute).Unix()
	}
	useMtime := func(sec int64) int64 {
		if !trimBuildCache {
			return sec
		}
		return time.Unix(sec, 0).Truncate(time.Minute).Add(-30 * time.Second).Unix()
	}

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
	mtime := writeMtime(now)
	checkTime(fmt.Sprintf("%x-a", id), mtime)
	checkTime(fmt.Sprintf("%x-d", entry.OutputID), mtime)

	// Get should not change recent mtimes.
	now = start + 10
	c.Get(id)
	checkTime(fmt.Sprintf("%x-a", id), mtime)
	checkTime(fmt.Sprintf("%x-d", entry.OutputID), mtime)

	// Get should change distant mtimes.
	now = start + 5000
	mtime2 := useMtime(now)
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
	checkTime(fmt.Sprintf("%x-a", dummyID(2)), writeMtime(start))

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
	checkTime(fmt.Sprintf("%x-a", dummyID(2)), writeMtime(start))
	if err := c.Trim(); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Get(id); err != nil {
		t.Fatal(err)
	}
	c.OutputFile(entry.OutputID)
	mtime3 := useMtime(now)
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

const testStart = 1020000000

type testCache struct {
	*DiskCache
	t            *testing.T
	start, clock time.Time
	outs         map[int]OutputID
}

func skipIfLockUnsupported(t *testing.T) {
	t.Helper()
	f, err := lockedfile.Edit(filepath.Join(t.TempDir(), "lockprobe"))
	if err != nil {
		if testenv.SyscallIsNotSupported(err) {
			t.Skipf("skipping: file locking unsupported (%v)", err)
		}
		t.Fatal(err)
	}
	f.Close()
}

func newTestCache(t *testing.T, dir string, stride, u int64) *testCache {
	t.Helper()
	d, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	c := &testCache{DiskCache: d, t: t, start: time.Unix(testStart, 0), outs: map[int]OutputID{}}
	c.clock = c.start
	d.now = func() time.Time { return c.clock }
	d.statfs = func(string) (avail, total, bsize uint64) { return 0, 0, 0 } // never low on disk
	d.openTime = c.clock
	d.trimBuildCache, d.sizeSample = true, stride
	d.sampleAcc.Store(u)
	return c
}

func (c *testCache) must(err error) {
	c.t.Helper()
	if err != nil {
		c.t.Fatal(err)
	}
}

func payload(n, size int) []byte {
	b := bytes.Repeat([]byte{'.'}, size)
	copy(b, fmt.Sprint(n, ":"))
	return b
}

// put and putExe store entries of total logical size (output + entrySize).
// A total of 0 stores an empty output.
func (c *testCache) put(n, total int) {
	c.t.Helper()
	out, _, err := c.Put(dummyID(n), bytes.NewReader(payload(n, max(0, total-entrySize))))
	c.must(err)
	c.outs[n] = out
}

func (c *testCache) putExe(n, total int) {
	c.t.Helper()
	out, _, err := c.PutExecutable(dummyID(n), "prog", bytes.NewReader(payload(n, max(0, total-entrySize))))
	c.must(err)
	c.outs[n] = out
}

func (c *testCache) trimMidScan(fn func()) {
	c.t.Helper()
	orig, calls := c.now, 0
	c.now = func() time.Time {
		calls++
		if calls == 2 {
			fn()
		}
		return c.clock
	}
	defer func() { c.now = orig }()
	c.must(c.Trim())
	if calls < 2 {
		c.t.Fatalf("Trim did not walk (calls=%d)", calls)
	}
}

func (c *testCache) wantEntries(present ...int) {
	c.t.Helper()
	live := map[OutputID]bool{}
	for _, n := range present {
		live[c.outs[n]] = true
	}
	for n, out := range c.outs {
		if _, err := os.Stat(c.fileName(dummyID(n), "a")); (err == nil) != slices.Contains(present, n) {
			c.t.Errorf("entry %d (-a): err=%v, want %v", n, err, slices.Contains(present, n))
		}
		if _, err := os.Stat(c.fileName(out, "d")); (err == nil) != live[out] {
			c.t.Errorf("entry %d (-d): err=%v, want %v", n, err, live[out])
		}
	}
}

func writeFile(t *testing.T, dir, name, data string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o666); err != nil {
		t.Fatal(err)
	}
}

func wantFile(t *testing.T, dir, name, want string) {
	t.Helper()
	if got, err := os.ReadFile(filepath.Join(dir, name)); err != nil || string(got) != want {
		t.Errorf("%s = %q, %v; want %q", name, got, err, want)
	}
}

func wantNoFiles(t *testing.T, dir string, names ...string) {
	t.Helper()
	for _, name := range names {
		if _, err := os.Stat(filepath.Join(dir, name)); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("stat %s: %v, want not exist", name, err)
		}
	}
}

func (c *testCache) use(n int) {
	c.t.Helper()
	e, err := c.Get(dummyID(n))
	c.must(err)
	c.OutputFile(e.OutputID)
}

func (c *testCache) wantMtime(want time.Time, paths ...string) {
	c.t.Helper()
	for _, p := range paths {
		info, err := os.Stat(p)
		c.must(err)
		if !info.ModTime().Equal(want) {
			c.t.Errorf("%s: mtime=%v, want %v", filepath.Base(p), info.ModTime(), want)
		}
	}
}

func TestCacheSize(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 1000, 0)

	// A flush below the stride creates nothing. The first Trim records the
	// measured size, and crossing a stride carries the remainder.
	c.put(1, 600)
	c.flush()
	wantNoFiles(t, dir, "size.txt", "trim.txt")

	c.must(c.Trim())
	wantFile(t, dir, "size.txt", "base 600\nestimated 600\n")
	wantFile(t, dir, "trim.txt", fmt.Sprint(c.clock.Unix()))

	c.put(2, 600) // 600+600=1200 -> flush 1000, carry 200
	c.flush()
	wantFile(t, dir, "size.txt", "base 600\nestimated 1600\n")
	if got := c.sampleAcc.Load(); got != 200 {
		t.Errorf("carry = %d, want 200", got)
	}

	// Only newly added bytes count: dup=0, shared=entrySize, truncated=size-oldSize.
	delta := func(want int64, fn func()) {
		t.Helper()
		b0 := c.sampleAcc.Load()
		fn()
		if got := c.sampleAcc.Load() - b0; got != want {
			t.Errorf("added %d, want %d", got, want)
		}
	}
	delta(0, func() { c.put(2, 600) })
	delta(entrySize, func() {
		out, _, err := c.Put(dummyID(3), bytes.NewReader(payload(2, 600-entrySize)))
		c.must(err)
		c.outs[3] = out
	})
	delta(600-entrySize-50, func() {
		c.must(os.WriteFile(c.fileName(c.outs[2], "d"), payload(0, 50), 0o666))
		c.put(2, 600)
	})

	// Trim drops expired entries, ignores stray files, counts executables by
	// contents, and replaces the estimate with the measured size.
	c.clock = c.start.Add(6 * 24 * time.Hour)
	c.put(4, 500)
	c.putExe(5, 800)
	stray := filepath.Join(dir, "00", "stray.tmp")
	writeFile(t, dir, filepath.Join("00", "stray.tmp"), strings.Repeat("x", 500))
	c.must(os.Chtimes(stray, c.start, c.start))
	writeFile(t, dir, "size.txt", "estimated 999999\nunknown 1\n")
	c.must(c.Trim())
	c.wantEntries(4, 5)
	if _, err := os.Stat(stray); err != nil {
		t.Errorf("stray removed: %v", err)
	}
	wantFile(t, dir, "size.txt", "base 1300\nestimated 1300\n")

	// Unbiased sampling across all offsets u in [0, stride), and split flushes
	// produce the same total as a single flush.
	const stride = 64
	c.sizeSample = stride
	for _, b := range []int64{0, 1, 17, 130} {
		var sum int64
		for u := range int64(stride) {
			c.sampleAcc.Store(u)
			c.addBytes(b)
			single := c.takeSampledAdded()
			sum += single

			c.sampleAcc.Store(u)
			var split int64
			for range b {
				c.addBytes(1)
				split += c.takeSampledAdded()
			}
			if split != single {
				t.Fatalf("u=%d B=%d: split=%d, want %d", u, b, split, single)
			}
		}
		if sum != b*stride {
			t.Errorf("B=%d: sum=%d, want %d", b, sum, b*stride)
		}
	}

	// trimDue handles corrupt data, exact 24h interval, and 1h future clock skew.
	now := c.clock
	for _, tc := range []struct {
		data string
		want bool
	}{
		{"bad", true},
		{fmt.Sprint(now.Add(-24*time.Hour + time.Second).Unix()), false},
		{fmt.Sprint(now.Add(-24 * time.Hour).Unix()), true},
		{fmt.Sprint(now.Add(59 * time.Minute).Unix()), false},
		{fmt.Sprint(now.Add(time.Hour).Unix()), true},
	} {
		if got := trimDue([]byte(tc.data), now, 24*time.Hour); got != tc.want {
			t.Errorf("trimDue(%q) = %v, want %v", tc.data, got, tc.want)
		}
	}
}

func TestCacheSizeFile(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 1000, 0)
	path := filepath.Join(dir, "size.txt")
	for _, tc := range []struct {
		in, want string // want "" means unchanged
	}{
		{in: "estimated 5000\ntorn", want: "base 0\nestimated 6000\n"},
		{in: "estimated -42\nestimated abc\nestimated 99999999999999999999\nestimated 5000 extra\nestimated 999"},
		{in: "estimated abc\nestimated   7000  \nestimated 9000\nfuture 1\n", want: "base 0\nestimated 8000\n"},
		{in: "estimated 9223372036854775807\n", want: fmt.Sprintf("base 0\nestimated %d\n", maxSizeBound)},
		{in: "estimated 1\nbase -5\nbase 7\nbase 9\n", want: "base 7\nestimated 1001\n"},
	} {
		writeFile(t, dir, "size.txt", tc.in)
		c.sampleAcc.Store(1000)
		c.flush()
		want := tc.want
		if want == "" {
			want = tc.in
		}
		wantFile(t, dir, "size.txt", want)
	}

	// writeSizeLocked always writes "base" then "estimated", so an untruncated
	// shorter rewrite never resurrects old values.
	writeSizeLocked(dir, cacheSizeState{base: 5e9, estimated: 6e9, hasEstimated: true})
	tmp := t.TempDir()
	writeSizeLocked(tmp, cacheSizeState{estimated: 0, hasEstimated: true})
	raw, err := os.ReadFile(filepath.Join(tmp, "size.txt"))
	c.must(err)
	f, err := os.OpenFile(path, os.O_WRONLY, 0o666)
	c.must(err)
	_, err = f.Write(raw)
	c.must(err)
	c.must(f.Close())
	if st, _, ok := readSize(dir); !ok || !st.hasEstimated || st.estimated != 0 || st.base != 0 {
		t.Errorf("torn clear: %+v, want estimated=0 base=0", st)
	}
}

func TestCleanCache(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	empty := t.TempDir()
	CleanCache(empty)
	wantNoFiles(t, empty, "size.txt", "trim.txt", "cache.log")

	dir := t.TempDir()
	writeFile(t, dir, "size.txt", "estimated 9876543210\nfuture 1\n")
	writeFile(t, dir, "trim.txt", "1234567890")
	before, err := os.Stat(filepath.Join(dir, "size.txt"))
	if err != nil {
		t.Fatal(err)
	}
	CleanCache(dir)
	if after, err := os.Stat(filepath.Join(dir, "size.txt")); err != nil || !os.SameFile(before, after) {
		t.Errorf("CleanCache replaced size.txt (%v)", err)
	}
	wantNoFiles(t, dir, "00")
	wantFile(t, dir, "size.txt", "base 0\nestimated 0\n")
	wantFile(t, dir, "trim.txt", "")
}

func TestCacheTrimDuringScan(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	// (a) A flush during a scan survives the post-scan size update.
	dir := t.TempDir()
	c := newTestCache(t, dir, 1<<40, 0)
	other := newTestCache(t, dir, 1000, 1000)
	c.put(1, 500)
	writeFile(t, dir, "size.txt", "estimated 0\n")
	c.trimMidScan(func() {
		other.flush()
		c.clock = c.clock.Add(2 * time.Second)
	})
	wantFile(t, dir, "size.txt", "base 500\nestimated 1500\n")
	wantFile(t, dir, "trim.txt", fmt.Sprint(c.clock.Unix()))
	if log, err := os.ReadFile(filepath.Join(dir, "cache.log")); err != nil || !strings.Contains(string(log), " in 2s, ") {
		t.Errorf("cache.log = %q, %v; want elapsed 2s", log, err)
	}

	// (b) CleanCache during a scan discards stale scan results.
	dir2 := t.TempDir()
	c2 := newTestCache(t, dir2, 1<<40, 0)
	c2.put(1, 500)
	writeFile(t, dir2, "size.txt", "estimated 5000\n")
	c2.trimMidScan(func() { CleanCache(dir2) })
	wantFile(t, dir2, "size.txt", "base 0\nestimated 0\n")
	wantFile(t, dir2, "trim.txt", "")
	if log, err := os.ReadFile(filepath.Join(dir2, "cache.log")); err != nil || strings.Contains(string(log), " trim:") {
		t.Errorf("cache.log = %q, %v; want no trim line", log, err)
	}

	// (c) An overtaking Trim causes the first scan to discard its stale results.
	dir3 := t.TempDir()
	c3 := newTestCache(t, dir3, 100, 0)
	c4 := newTestCache(t, dir3, 100, 0)
	c3.sizeMinTarget, c4.sizeMinTarget = 1000, 1000
	c3.put(1, 500)
	c3.clock = c3.start.Add(2 * time.Hour)
	c3.put(2, 700)
	writeFile(t, dir3, "trim.txt", fmt.Sprint(c3.start.Unix()))
	writeFile(t, dir3, "size.txt", "base 0\nestimated 10000\n")
	c3.sampleAcc.Store(100)
	c4.clock = c4.start.Add(4 * time.Hour)
	c3.trimMidScan(func() {
		c4.put(3, 600)
		c4.must(c4.Trim())
	})
	c3.wantEntries()
	c4.wantEntries(3)
	wantFile(t, dir3, "size.txt", "base 600\nestimated 600\n")
	wantFile(t, dir3, "trim.txt", fmt.Sprint(c4.clock.Unix()))
	log3, err := os.ReadFile(filepath.Join(dir3, "cache.log"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(log3), " trim: "); got != 1 {
		t.Errorf("cache.log has %d trim lines, want 1:\n%s", got, log3)
	}
}

func TestDefaultCacheAtExit(t *testing.T) {
	if os.Getenv("TEST_CACHE_ATEXIT") != "" {
		Default()
		c := defaultDiskCache.Load()
		c.trimBuildCache, c.sizeSample = true, 1000
		c.sampleAcc.Store(1000)
		base.Exit()
	}
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	writeFile(t, dir, "size.txt", "estimated 100\n")
	cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^"+t.Name()+"$")
	cmd.Env = append(os.Environ(), "GOCACHE="+dir, "GOCACHEPROG=", "GOENV=off", "TEST_CACHE_ATEXIT=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("subprocess: %v\n%s", err, out)
	}
	wantFile(t, dir, "size.txt", "base 0\nestimated 1100\n")
}

const logTimeFormat = "2006/01/02 15:04:05"

func TestCacheLog(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 1<<40, 0)
	c.put(1, 1<<20) // 1 MiB expired so before (0.001 GiB) != after (0.000 GiB)
	c.clock = c.start.Add(6 * 24 * time.Hour)
	c.put(2, 500)
	c.must(c.Trim())
	want := c.clock.Format(logTimeFormat) + " trim: 24h trigger: 0.001 GiB -> 0.000 GiB in 0s, removed 2 of 4 files, base=0.000 GiB next=5.000 GiB\n"
	wantFile(t, dir, "cache.log", want)

	// Routine sampled flushes and non-due trims are not logged.
	c.sampleAcc.Store(c.sizeSample)
	c.flush()
	c.must(c.Trim())
	wantFile(t, dir, "cache.log", want)

	// CleanCache preserves rotated logs and appends a clean line.
	writeFile(t, dir, "cache.1.log", "one\n")
	CleanCache(dir)
	wantFile(t, dir, "cache.1.log", "one\n")
	if log, err := os.ReadFile(filepath.Join(dir, "cache.log")); err != nil || !strings.HasSuffix(string(log), " clean -cache\n") {
		t.Errorf("cache.log = %q, %v; want clean -cache", log, err)
	}

	// A trim scan whose prior estimate drifted by >10% and >1 GiB logs the delta.
	const GiB int64 = 1 << 30
	driftDir := t.TempDir()
	dc := newTestCache(t, driftDir, 1<<40, 0)
	dc.put(1, 500)
	writeFile(t, driftDir, "size.txt", fmt.Sprintf("estimated %d\n", 2*GiB))
	dc.must(dc.Trim())
	if log, err := os.ReadFile(filepath.Join(driftDir, "cache.log")); err != nil ||
		!strings.Contains(string(log), " (prior estimate=2.000 GiB actual=0.000 GiB delta=") {
		t.Errorf("drifted trim cache.log = %q, %v; want prior estimate delta", log, err)
	}

	// Lines up to 2048 bytes are written intact, while longer lines are truncated.
	truncDir := t.TempDir()
	exactMsg := strings.Repeat("x", 2048-len(logTimeFormat)-2)
	appendLogLocked(truncDir, c.clock, "%s", exactMsg)
	wantExact := c.clock.Format(logTimeFormat) + " " + exactMsg + "\n"
	if len(wantExact) != 2048 {
		t.Fatalf("len(wantExact) = %d, want 2048", len(wantExact))
	}
	wantFile(t, truncDir, "cache.log", wantExact)

	appendLogLocked(truncDir, c.clock, "%s", exactMsg+"y")
	wantTrunc := (c.clock.Format(logTimeFormat) + " " + exactMsg + "y\n")[:2048] + " [truncated]\n"
	wantFile(t, truncDir, "cache.log", wantExact+wantTrunc)

	for _, tc := range []struct {
		est, actual int64
		want        string
	}{
		{22 * GiB, 20 * GiB, ""},
		{22*GiB + 1, 20 * GiB, " (prior estimate=22.000 GiB actual=20.000 GiB delta=+10.0%)"},
		{1 * GiB, 0, ""},
		{1*GiB + 1, 0, " (prior estimate=1.000 GiB actual=0.000 GiB)"},
		{8 * GiB, 10 * GiB, " (prior estimate=8.000 GiB actual=10.000 GiB delta=-20.0%)"},
	} {
		if got := formatEstimateDelta(tc.est, tc.actual); got != tc.want {
			t.Errorf("formatEstimateDelta(%d, %d) = %q, want %q", tc.est, tc.actual, got, tc.want)
		}
	}
}

func TestCacheLogRotation(t *testing.T) {
	t.Parallel()

	const roll = 128 << 10
	now := time.Date(2026, 1, 2, 15, 4, 5, 0, time.Local)
	line := func(msg string) string { return now.Format(logTimeFormat) + " " + msg + "\n" }

	dir := t.TempDir()
	pre := strings.Repeat("a", roll-1)
	full := strings.Repeat("b", roll)

	// Below logRollSize does not roll, and reaching it rolls cache.log to cache.1.log.
	writeFile(t, dir, "cache.log", pre)
	appendLogLocked(dir, now, "first")
	wantNoFiles(t, dir, "cache.1.log")
	appendLogLocked(dir, now, "second")
	wantFile(t, dir, "cache.1.log", pre+line("first"))

	// A subsequent roll shifts cache.1.log to cache.2.log.
	writeFile(t, dir, "cache.log", full)
	appendLogLocked(dir, now, "third")
	wantFile(t, dir, "cache.2.log", pre+line("first"))

	// A missing cache.1.log heals the gap without clobbering cache.2.log.
	// On Plan 9, rename removes dst even if src does not exist.
	if runtime.GOOS != "plan9" {
		if err := os.Remove(filepath.Join(dir, "cache.1.log")); err != nil {
			t.Fatal(err)
		}
		writeFile(t, dir, "cache.log", full)
		appendLogLocked(dir, now, "fourth")
		wantFile(t, dir, "cache.2.log", pre+line("first"))
		wantFile(t, dir, "cache.1.log", full)
	}

	// Repeated rotation failures allow writes up to 2*logRollSize, then halt logging.
	stuck := t.TempDir()
	if err := os.MkdirAll(filepath.Join(stuck, "cache.2.log", "x"), 0o777); err != nil {
		t.Fatal(err)
	}
	writeFile(t, stuck, "cache.1.log", "old\n")
	nearCap := strings.Repeat("x", 2*roll-10)
	writeFile(t, stuck, "cache.log", nearCap)
	appendLogLocked(stuck, now, "allowed")
	appendLogLocked(stuck, now, "halted")
	wantFile(t, stuck, "cache.log", nearCap+line("allowed"))
	wantFile(t, stuck, "cache.1.log", "old\n")
}

func TestCacheTargetSize(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		mt   time.Time
		want bool
	}{
		{time.Unix(testStart, 0), true},
		{time.Unix(testStart, 1), false},
		{time.Unix(testStart+30, 0), false},
	} {
		if got := isYoungGen(tc.mt); got != tc.want {
			t.Errorf("isYoungGen(%v) = %v, want %v", tc.mt, got, tc.want)
		}
	}

	const GiB int64 = 1 << 30
	c := &DiskCache{sizeMinTarget: defaultMinTarget}
	for _, tc := range []struct {
		base, want int64
	}{
		{0, 5 * GiB},
		{3 * GiB, 5 * GiB},
		{4 * GiB, 6 * GiB},
		{20 * GiB, 30 * GiB},
		{20*GiB + 4, 25*GiB + 5},
		{24 * GiB, 30 * GiB},
		{100 * GiB, 125 * GiB},
		{maxSizeBound, maxSizeBound + maxSizeBound/4},
	} {
		if got := c.targetSize(tc.base); got != tc.want {
			t.Errorf("targetSize(%d) = %d, want %d", tc.base, got, tc.want)
		}
	}
}

func TestCacheGenerations(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		put  func(*testCache)
	}{
		{"regular", func(c *testCache) { c.put(1, 500) }},
		{"executable", func(c *testCache) { c.putExe(1, 600) }},
		{"empty", func(c *testCache) { c.put(1, 0) }},
		{"empty_executable", func(c *testCache) { c.putExe(1, 0) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestCache(t, t.TempDir(), 1<<40, 0)
			c.clock = c.start.Add(45 * time.Second)
			tc.put(c)
			f := []string{c.fileName(dummyID(1), "a"), c.fileName(c.outs[1], "d")}
			c.wantMtime(c.start, f...)

			// Reuse before 30m keeps the young generation mark unchanged.
			c.clock = c.start.Add(29*time.Minute + 59*time.Second)
			c.use(1)
			c.wantMtime(c.start, f...)

			// Reuse at or after 30m promotes to tenured30m (:30).
			c.clock = c.start.Add(30*time.Minute + 20*time.Second)
			c.use(1)
			t1 := c.start.Add(29*time.Minute + 30*time.Second)
			c.wantMtime(t1, f...)

			// Tenured entries refresh at most once per hour.
			c.clock = t1.Add(59*time.Minute + 59*time.Second)
			c.use(1)
			c.wantMtime(t1, f...)

			c.clock = t1.Add(time.Hour)
			c.use(1)
			c.wantMtime(c.clock.Truncate(time.Minute).Add(-30*time.Second), f...)
		})
	}

	// Rewriting an existing entry counts as a use, while concurrent cold writes
	// that have not reached Chtimes yet are not falsely tenured.
	c := newTestCache(t, t.TempDir(), 1<<40, 0)
	c.clock = c.start.Add(40 * time.Second)
	c.put(1, 500)
	f1 := []string{c.fileName(dummyID(1), "a"), c.fileName(c.outs[1], "d")}

	c.clock = c.start.Add(20 * time.Minute)
	c.put(1, 500)
	c.wantMtime(c.start, f1...)

	c.clock = c.start.Add(40*time.Minute + 15*time.Second)
	c.put(1, 500)
	c.wantMtime(c.clock.Truncate(time.Minute).Add(-30*time.Second), f1...)

	// A new action sharing an existing output marks the new index entry young
	// and refreshes the shared output as tenured30m.
	c.clock = c.start.Add(2*time.Hour + 15*time.Second)
	m2 := c.clock.Truncate(time.Minute)
	_, _, err := c.Put(dummyID(2), bytes.NewReader(payload(1, 500-entrySize)))
	c.must(err)
	c.wantMtime(m2, c.fileName(dummyID(2), "a"))
	c.wantMtime(m2.Add(-30*time.Second), c.fileName(c.outs[1], "d"))

	// An index file whose raw OS mtime is newer than tenured stays young on rewrite.
	c.put(3, 500)
	a3 := c.fileName(dummyID(3), "a")
	c.must(os.Chtimes(a3, m2.Add(10*time.Second), m2.Add(10*time.Second)))
	c.put(3, 500)
	c.wantMtime(m2, a3)
}

func TestCacheTrimYoungGen(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 100, 0)
	c.sizeMinTarget = 100
	c.clock = c.start.Add(40 * time.Second)
	c.put(1, 400)
	c.putExe(2, 500)
	c.put(3, 0)
	c.put(4, 300)

	// Entries 1 to 3 are used before 30m (stay young). Entry 4 is used at 30m (tenured).
	c.clock = c.start.Add(29*time.Minute + 59*time.Second)
	for _, id := range []int{1, 2, 3} {
		c.use(id)
	}
	c.clock = c.start.Add(30 * time.Minute)
	c.use(4)

	seedOver := func() {
		writeFile(t, dir, "trim.txt", fmt.Sprint(c.clock.Add(-2*time.Hour).Unix()))
		writeFile(t, dir, "size.txt", "base 0\nestimated 10000\n")
		c.sampleAcc.Store(100)
	}

	// Just before 90m, a size triggered trim keeps all young entries and counts them in base.
	c.clock = c.start.Add(89*time.Minute + 59*time.Second)
	seedOver()
	c.must(c.Trim())
	c.wantEntries(1, 2, 3, 4)
	wantFile(t, dir, "size.txt", fmt.Sprintf("base %d\nestimated %[1]d\n", 400+500+entrySize+300))

	// At 90m, a size triggered trim removes young entries 1 to 3 and keeps tenured entry 4.
	c.clock = c.start.Add(90 * time.Minute)
	seedOver()
	c.must(c.Trim())
	c.wantEntries(4)
	wantFile(t, dir, "size.txt", "base 300\nestimated 300\n")

	// Daily trim (added==0) never deletes young entries early even over target,
	// treats plain mtimes as tenured, and expires both generations at 5d+1h.
	dir2 := t.TempDir()
	c2 := newTestCache(t, dir2, 100, 0)
	c2.clock = c2.start.Add(10 * 24 * time.Hour)
	now := c2.clock
	exp := now.Add(-5*24*time.Hour - time.Hour - time.Minute)
	mtimes := []time.Time{
		now.Add(-24 * time.Hour), // young, 1d old: kept, excluded from base
		exp,                      // young, >5d+1h: expired
		now.Add(-4*24*time.Hour + 16*time.Second), // plain mtime (tenured), 4d old: kept in base
		exp.Add(-30 * time.Second),                // tenured, >5d+1h: expired
	}
	for i, mt := range mtimes {
		c2.put(i+1, 400+(i/2)*100)
		c2.must(os.Chtimes(c2.fileName(dummyID(i+1), "a"), mt, mt))
		c2.must(os.Chtimes(c2.fileName(c2.outs[i+1], "d"), mt, mt))
	}
	writeFile(t, dir2, "trim.txt", fmt.Sprint(now.Add(-24*time.Hour).Unix()))
	writeFile(t, dir2, "size.txt", "base 100\nestimated 99999\n")
	c2.sampleAcc.Store(0)
	c2.must(c2.Trim())
	c2.wantEntries(1, 3)
	wantFile(t, dir2, "size.txt", "base 500\nestimated 900\n")
}

func TestCacheTrimSize(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	// Real Put and Trim lifecycle with no seeded size.txt or trim.txt (1000 B stands for 5 GiB).
	dir := t.TempDir()
	c := newTestCache(t, dir, 100, 0)
	c.sizeMinTarget = 1000

	// 0h: the first Trim (a daily trim) records the cold build as the baseline.
	c.put(1, 300) // shared deps
	c.put(2, 300) // early edit
	c.must(c.Trim())
	wantFile(t, dir, "size.txt", "base 600\nestimated 600\n")

	// 35m: reuse entry 1 (promotes to tenured30m) and add entry 3.
	c.clock = c.start.Add(35 * time.Minute)
	c.use(1)
	c.put(3, 300)
	c.flush()
	wantFile(t, dir, "size.txt", "base 600\nestimated 900\n")

	// 2h: crossing targetSize(600) = sizeMinTarget (900 -> 1600) triggers the first
	// size trim, removing 2h old young entry 2 and keeping 85m old entry 3 in base.
	c.clock = c.start.Add(2 * time.Hour)
	c.put(4, 700)
	c.must(c.Trim())
	c.wantEntries(1, 3, 4)
	wantFile(t, dir, "size.txt", "base 1300\nestimated 1300\n")
	if log, err := os.ReadFile(filepath.Join(dir, "cache.log")); err != nil || !strings.Contains(string(log), " trim: size trigger: ") {
		t.Errorf("cache.log = %q, %v; want size trigger", log, err)
	}

	// Crossing targetSize(1300) = 1950 right away waits for the 1h spacing.
	c.put(5, 900)
	c.must(c.Trim())
	c.wantEntries(1, 3, 4, 5)
	wantFile(t, dir, "size.txt", "base 1300\nestimated 2200\n")

	// Less than 1h spacing blocks a size trim, and an unsampled Trim at 3h is ignored.
	c.clock = c.start.Add(3*time.Hour - time.Second)
	c.put(6, 1200)
	c.must(c.Trim())
	c.clock = c.start.Add(3 * time.Hour)
	c.must(c.Trim())
	c.wantEntries(1, 3, 4, 5, 6)

	// A sampled put at 3h (estimate 3600 >= 1950) reclaims 3 but keeps 1h old 4 and 5.
	c.put(7, 200)
	c.must(c.Trim())
	c.wantEntries(1, 4, 5, 6, 7)
	wantFile(t, dir, "size.txt", "base 3300\nestimated 3300\n")

	// Next day (28h): daily trim lowers base to tenured deps (300) without removing 4 to 7.
	// An hour later, a small sampled put crosses targetSize(300) = 1000 and reclaims 4 to 7.
	c.clock = c.start.Add(28 * time.Hour)
	c.must(c.Trim())
	c.wantEntries(1, 4, 5, 6, 7)
	wantFile(t, dir, "size.txt", "base 300\nestimated 3300\n")

	c.clock = c.start.Add(29 * time.Hour)
	c.put(8, 200)
	c.must(c.Trim())
	c.wantEntries(1, 8)
	wantFile(t, dir, "size.txt", "base 500\nestimated 500\n")

	// Base follows the cache back down: after a 5d+1h expiry, the daily trim records
	// the smaller baseline, so the next target is sizeMinTarget again.
	dir2 := t.TempDir()
	c2 := newTestCache(t, dir2, 200, 0)
	c2.sizeMinTarget = 1000
	c2.put(1, 900)
	c2.must(c2.Trim())
	wantFile(t, dir2, "size.txt", "base 900\nestimated 900\n")
	c2.clock = c2.start.Add(6 * 24 * time.Hour)
	c2.put(2, 200)
	c2.must(c2.Trim())
	c2.wantEntries(2)
	wantFile(t, dir2, "size.txt", "base 200\nestimated 200\n")
}

func TestCacheCloseKeepsOwnOutput(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 100, 0)
	c.sizeMinTarget = 1000
	c.put(1, 500)

	// Open a long command at 90m, write entry 2, and Close 2h later.
	// Close measures youngGenDeleteAge from openTime, so entry 2 is kept while entry 1 is removed.
	c.clock = c.start.Add(90 * time.Minute)
	c.openTime = c.clock
	c.put(2, 800)

	c.clock = c.start.Add(90*time.Minute + 2*time.Hour)
	writeFile(t, dir, "trim.txt", fmt.Sprint(c.clock.Add(-2*time.Hour).Unix()))
	writeFile(t, dir, "size.txt", "base 0\nestimated 10000\n")
	c.sampleAcc.Store(100)
	c.must(c.Close())
	c.wantEntries(2)
	wantFile(t, dir, "size.txt", "base 0\nestimated 800\n")
}

func TestCacheLowDiskFree(t *testing.T) {
	const MiB, GiB = 1 << 20, 1 << 30
	for _, tc := range []struct {
		avail, total, bsize uint64
		want                bool
	}{
		{512*MiB/4096 - 1, 20 * GiB / 4096, 4096, true}, // 20 GiB: 512 MiB is smaller than 5%
		{512 * MiB / 4096, 20 * GiB / 4096, 4096, false},
		{512*MiB/4096 - 1, 10 * GiB / 4096, 4096, true}, // 10 GiB: 512 MiB is 5%
		{50 * MiB / 4096, 1 * GiB / 4096, 4096, true},   // 1 GiB: 5% is about 51 MiB
		{52 * MiB / 4096, 1 * GiB / 4096, 4096, false},
		{600 * MiB / 1024, 2048 * GiB / 1024, 1024, false}, // 2 TiB
		{0, GiB, 1, true},   // completely full
		{0, 0, 4096, false}, // unknown (statfs error, tmpfs without a size limit, non-Linux)
		{0, GiB, 0, false},
		{20, 10, 4096, false}, // avail > total
		{0, ^uint64(0), 1 << 30, true},
		{0, ^uint64(0), 1 << 40, true},
		{1, ^uint64(0), 1 << 40, false},
	} {
		c := &DiskCache{statfs: func(string) (avail, total, bsize uint64) { return tc.avail, tc.total, tc.bsize }}
		free, low := c.lowDiskFree()
		if low != tc.want || low && free != int64(tc.avail*tc.bsize) {
			t.Errorf("lowDiskFree() with avail=%d total=%d bsize=%d = %d, %v; want low=%v", tc.avail, tc.total, tc.bsize, free, low, tc.want)
		}
	}
}

func TestCacheTrimLowDisk(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 100, 0)
	low := false
	c.statfs = func(string) (avail, total, bsize uint64) {
		if low {
			return 1, 100, 1 << 20 // 1 MiB of 100 MiB available
		}
		return 50, 100, 1 << 20
	}
	c.clock = c.start.Add(10 * time.Hour)
	now := c.clock
	for i, mt := range []time.Time{
		now.Add(-90 * time.Minute),                // young, 90m: removed
		now.Add(-89 * time.Minute),                // young, 89m: kept in base
		now.Add(-90*time.Minute - 30*time.Second), // tenured, 90m30s: removed
		now.Add(-89*time.Minute - 30*time.Second), // tenured, 89m30s: kept in base
		now.Add(-3*time.Hour + 16*time.Second),    // plain mtime (tenured), ~3h: removed
	} {
		c.put(i+1, 100*(i+1))
		c.must(os.Chtimes(c.fileName(dummyID(i+1), "a"), mt, mt))
		c.must(os.Chtimes(c.fileName(c.outs[i+1], "d"), mt, mt))
	}
	c.sampleAcc.Store(0)
	writeFile(t, dir, "size.txt", "base 0\nestimated 1500\n")

	// A low disk trim waits for minTrimSpacing (with or without sampled bytes),
	// and a disk with space left does not trim.
	writeFile(t, dir, "trim.txt", fmt.Sprint(now.Add(-minTrimSpacing+time.Second).Unix()))
	low = true
	c.must(c.Trim())
	c.sampleAcc.Store(100)
	c.must(c.Trim())
	writeFile(t, dir, "trim.txt", fmt.Sprint(now.Add(-minTrimSpacing).Unix()))
	low = false
	c.must(c.Trim())
	c.wantEntries(1, 2, 3, 4, 5)

	// With no bytes added, a nearly full disk removes all entries at least 90m old
	// regardless of generation.
	low = true
	c.must(c.Trim())
	c.wantEntries(2, 4)
	wantFile(t, dir, "size.txt", "base 600\nestimated 600\n")
	wantFile(t, dir, "cache.log", now.Format(logTimeFormat)+
		" trim: low disk trigger: 0.000 GiB -> 0.000 GiB in 0s, removed 6 of 10 files,"+
		" base=0.000 GiB next=5.000 GiB free=0.001 GiB (consider providing more space)\n")

	// Later, Close measures age from openTime, so it keeps tenured entry 6
	// (over 90m old now but not at openTime) and the new entry 7 it just wrote.
	c.clock = now.Add(minTrimSpacing)
	c.openTime = c.clock.Add(-30 * time.Minute)
	c.put(6, 100)
	mt := c.clock.Add(-100*time.Minute - 30*time.Second)
	c.must(os.Chtimes(c.fileName(dummyID(6), "a"), mt, mt))
	c.must(os.Chtimes(c.fileName(c.outs[6], "d"), mt, mt))
	c.put(7, 100)
	c.must(c.Close())
	c.wantEntries(6, 7)
}

func TestCacheTrimBuildCacheDisabled(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 100, 0)
	c.trimBuildCache = false
	c.sizeMinTarget = 100

	c.put(1, 500) // :00 mtime (looks like young gen)
	c.clock = c.start.Add(40 * time.Second)
	c.put(2, 500) // raw non-minute mtime
	c.flush()
	if got := c.sampleAcc.Load(); got != 0 {
		t.Errorf("sampleAcc = %d, want 0", got)
	}
	wantNoFiles(t, dir, "size.txt", "trim.txt", "cache.log")
	c.wantMtime(c.clock, c.fileName(dummyID(2), "a"), c.fileName(c.outs[2], "d"))

	// Use at 35m does not trigger 30m promotion (legacy uses 1h).
	c.clock = c.start.Add(35 * time.Minute)
	c.use(1)
	c.wantMtime(c.start, c.fileName(dummyID(1), "a"), c.fileName(c.outs[1], "d"))

	// Trim uses the 5 day legacy rule and does not delete young entries early at 2h.
	c.clock = c.start.Add(2 * time.Hour)
	c.must(c.Trim())
	c.wantEntries(1, 2)
	wantNoFiles(t, dir, "size.txt", "cache.log")

	c.clock = c.start.Add(6 * 24 * time.Hour)
	c.must(c.Trim())
	c.wantEntries()
	wantNoFiles(t, dir, "size.txt", "cache.log")
}

func TestCacheRefreshUsed(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	other := newTestCache(t, dir, 100, 0)
	long := newTestCache(t, dir, 100, 0)
	other.sizeMinTarget, long.sizeMinTarget = 100, 100

	// Another command populates entries 1 (regular), 2 (executable), 3 (unused), and 5 (tenured).
	other.put(1, 400)
	other.putExe(2, 500)
	other.put(3, 400)
	other.put(5, 400)
	f12 := []string{
		other.fileName(dummyID(1), "a"), other.fileName(other.outs[1], "d"),
		other.fileName(dummyID(2), "a"), other.fileName(other.outs[2], "d"),
	}
	f5 := []string{other.fileName(dummyID(5), "a"), other.fileName(other.outs[5], "d")}

	// Worst case young hit at 29m59s (not promoted on access), including a Get-only
	// executable entry whose -d directory must be protected without OutputFile.
	long.clock = long.start.Add(29*time.Minute + 59*time.Second)
	long.use(1)
	if _, err := long.Get(dummyID(2)); err != nil {
		t.Fatal(err)
	}
	long.wantMtime(long.start, f12...)

	// Entry 5 is tenured by reuse at 30m and then accessed by the long command at 35m.
	other.clock = other.start.Add(30 * time.Minute)
	other.use(5)
	long.clock = long.start.Add(35 * time.Minute)
	long.use(5)
	long.wantMtime(long.start.Add(29*time.Minute+30*time.Second), f5...)

	// Fresh Put 1m before tick 1 stays young and is refreshed once mtime reaches 60m (before 80m).
	long.clock = long.start.Add(48*time.Minute + 59*time.Second)
	long.put(4, 400)
	f4 := []string{long.fileName(dummyID(4), "a"), long.fileName(long.outs[4], "d")}

	// Tick 1 (49m59s): young entries under 60m are left alone so reuse in [30m, 60m) still tenures.
	long.clock = long.start.Add(49*time.Minute + 59*time.Second)
	long.refreshUsed()
	long.wantMtime(long.start, f12...)
	long.wantMtime(long.start.Add(48*time.Minute), f4...)

	// Tick 2 (69m59s): f12 refreshes to :00 (stays young). Tick 3 (89m59s): tenured f5 refreshes to :30.
	long.clock = long.start.Add(69*time.Minute + 59*time.Second)
	long.refreshUsed()
	long.wantMtime(long.start.Add(69*time.Minute), f12...)
	long.wantMtime(long.start.Add(48*time.Minute), f4...)

	long.clock = long.start.Add(89*time.Minute + 59*time.Second)
	long.refreshUsed()
	long.wantMtime(long.start.Add(88*time.Minute+30*time.Second), f5...)
	long.wantMtime(long.start.Add(48*time.Minute), f4...)

	// Tick 4 (109m59s): fresh entry 4 reaches 60m and refreshes to :00 (stays young).
	long.clock = long.start.Add(109*time.Minute + 59*time.Second)
	long.refreshUsed()
	long.wantMtime(long.start.Add(109*time.Minute), f4...)

	// A concurrent size triggered trim past 90m removes unused entry 3 while keeping
	// all entries accessed by the long running command.
	other.clock = other.start.Add(2 * time.Hour)
	writeFile(t, dir, "trim.txt", fmt.Sprint(other.clock.Add(-2*time.Hour).Unix()))
	writeFile(t, dir, "size.txt", "base 0\nestimated 10000\n")
	other.sampleAcc.Store(100)
	other.must(other.Trim())
	other.wantEntries(1, 2, 5)
	long.wantEntries(4)

	// Ongoing refreshes keep accessed entries alive across a 6 day daily trim (5d+1h).
	for day := 1; day <= 6; day++ {
		long.clock = long.start.Add(time.Duration(day) * 24 * time.Hour)
		long.refreshUsed()
	}
	other.clock = other.start.Add(6 * 24 * time.Hour)
	other.sampleAcc.Store(0)
	other.must(other.Trim())
	other.wantEntries(1, 2, 5)
	long.wantEntries(4)

	// Once the long command stops refreshing, young entries 1, 2, and 4 were never tenured
	// and are reclaimed by a later size triggered trim, while tenured entry 5 survives.
	other.clock = other.start.Add(6*24*time.Hour + 2*time.Hour)
	writeFile(t, dir, "size.txt", "base 0\nestimated 10000\n")
	other.sampleAcc.Store(100)
	other.must(other.Trim())
	other.wantEntries(5)
	long.wantEntries()

	// With trimBuildCache disabled, accessed entries are not tracked or refreshed.
	off := newTestCache(t, t.TempDir(), 100, 0)
	off.trimBuildCache = false
	off.put(1, 500)
	off.use(1)
	off.used.Range(func(k, _ any) bool {
		t.Errorf("disabled cache recorded used entry %v", k)
		return false
	})
	off.clock = off.start.Add(70 * time.Minute)
	off.refreshUsed()
	off.wantMtime(off.start, off.fileName(dummyID(1), "a"), off.fileName(off.outs[1], "d"))
}

func TestCacheTrimParallel(t *testing.T) {
	skipIfLockUnsupported(t)
	t.Parallel()

	dir := t.TempDir()
	c := newTestCache(t, dir, 1<<40, 0)
	put := func(n int) {
		if n%4 == 0 {
			c.putExe(n, 200)
		} else {
			c.put(n, 200)
		}
	}

	// Put one expired and one fresh entry in each of the 256 subdirectories.
	for i := range 256 {
		put(i)
	}
	c.clock = c.start.Add(6 * 24 * time.Hour)
	fresh := make([]int, 256)
	for i := range 256 {
		fresh[i] = 256 + i
		put(256 + i)
	}

	c.must(c.Trim())
	c.wantEntries(fresh...)
	wantFile(t, dir, "size.txt", "base 51200\nestimated 51200\n")
	if log, err := os.ReadFile(filepath.Join(dir, "cache.log")); err != nil || !strings.Contains(string(log), " removed 512 of 1024 files,") {
		t.Errorf("cache.log = %q, %v; want removed 512 of 1024 files", log, err)
	}
}

func TestPutOptionalLowDisk(t *testing.T) {
	t.Parallel()

	c := newTestCache(t, t.TempDir(), 100, 0)
	low := true
	statfsCalls := 0
	c.statfs = func(string) (avail, total, bsize uint64) {
		statfsCalls++
		if low {
			return 1, 100, 1 << 20 // 1 MiB of 100 MiB available
		}
		return 50, 100, 1 << 20
	}

	// On a nearly full disk, PutOptional returns the output ID but stores nothing.
	out, size, err := PutOptional(c, dummyID(1), bytes.NewReader([]byte("abc")))
	if err != nil || size != 3 || out != OutputID(sha256.Sum256([]byte("abc"))) {
		t.Fatalf("PutOptional = %x, %d, %v", out, size, err)
	}
	if _, err := os.Stat(c.fileName(dummyID(1), "a")); err == nil {
		t.Errorf("PutOptional stored an index entry on a nearly full disk")
	}
	if _, err := os.Stat(c.fileName(out, "d")); err == nil {
		t.Errorf("PutOptional stored an output on a nearly full disk")
	}
	if got := c.sampleAcc.Load(); got != 0 {
		t.Errorf("sampleAcc = %d, want 0", got)
	}

	// Put and PutExecutable still store their outputs.
	out, _, err = c.Put(dummyID(2), bytes.NewReader([]byte("def")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(c.OutputFile(out)); err != nil {
		t.Errorf("Put did not store output: %v", err)
	}
	if _, err := c.Get(dummyID(2)); err != nil {
		t.Errorf("Get after Put failed: %v", err)
	}
	out, _, err = c.PutExecutable(dummyID(3), "prog", bytes.NewReader([]byte("ghi")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(c.OutputFile(out)); err != nil {
		t.Errorf("PutExecutable did not store output: %v", err)
	}

	// The check runs once, so PutOptional keeps skipping even if space is freed later.
	low = false
	if _, _, err := PutOptional(c, dummyID(4), bytes.NewReader([]byte("jkl"))); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(c.fileName(dummyID(4), "a")); err == nil {
		t.Errorf("PutOptional stored an index entry after the disk was nearly full")
	}
	if statfsCalls != 1 {
		t.Errorf("statfs called %d times, want 1", statfsCalls)
	}

	// With enough free space, PutOptional stores its output.
	ok := newTestCache(t, t.TempDir(), 100, 0)
	ok.statfs = func(string) (avail, total, bsize uint64) { return 50, 100, 1 << 20 }
	out, _, err = PutOptional(ok, dummyID(5), bytes.NewReader([]byte("mno")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(ok.OutputFile(out)); err != nil {
		t.Errorf("PutOptional with free space did not store output: %v", err)
	}

	// With trimBuildCache disabled, PutOptional writes without calling statfs.
	off := newTestCache(t, t.TempDir(), 100, 0)
	off.trimBuildCache = false
	off.statfs = func(string) (avail, total, bsize uint64) {
		t.Errorf("statfs called with trimBuildCache disabled")
		return 1, 100, 1 << 20
	}
	out, _, err = PutOptional(off, dummyID(6), bytes.NewReader([]byte("pqr")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(off.OutputFile(out)); err != nil {
		t.Errorf("PutOptional with trimBuildCache disabled did not store output: %v", err)
	}
}
