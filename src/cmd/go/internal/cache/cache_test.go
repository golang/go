// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
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
	c.sampleAcc.Store(0)
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
	wantFile(t, dir, "size.txt", "estimated 600\n")
	wantFile(t, dir, "trim.txt", fmt.Sprint(c.clock.Unix()))

	c.put(2, 600) // 600+600=1200 -> flush 1000, carry 200
	c.flush()
	wantFile(t, dir, "size.txt", "estimated 1600\n")
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
	wantFile(t, dir, "size.txt", "estimated 1300\n")

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
	for _, tc := range []struct {
		in, want string // want "" means unchanged
	}{
		{in: "estimated 5000\ntorn", want: "estimated 6000\n"},
		{in: "estimated -42\nestimated abc\nestimated 99999999999999999999\nestimated 5000 extra\nestimated 999"},
		{in: "estimated abc\nestimated   7000  \nestimated 9000\nfuture 1\n", want: "estimated 8000\n"},
		{in: "estimated 9223372036854775807\n", want: fmt.Sprintf("estimated %d\n", maxSizeBound)},
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
	wantFile(t, dir, "size.txt", "estimated 0\n")
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
	wantFile(t, dir, "size.txt", "estimated 1500\n")
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
	wantFile(t, dir2, "size.txt", "estimated 0\n")
	wantFile(t, dir2, "trim.txt", "")
	if log, err := os.ReadFile(filepath.Join(dir2, "cache.log")); err != nil || strings.Contains(string(log), " trim:") {
		t.Errorf("cache.log = %q, %v; want no trim line", log, err)
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
	wantFile(t, dir, "size.txt", "estimated 1100\n")
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
	want := c.clock.Format(logTimeFormat) + " trim: 24h trigger: 0.001 GiB -> 0.000 GiB in 0s, removed 2 of 4 files\n"
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
