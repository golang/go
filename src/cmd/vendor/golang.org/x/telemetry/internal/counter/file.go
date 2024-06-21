// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package counter

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/telemetry/internal/mmap"
	"golang.org/x/telemetry/internal/telemetry"
)

// A file is a counter file.
type file struct {
	// Linked list of all known counters.
	// (Linked list insertion is easy to make lock-free,
	// and we don't want the initial counters incremented
	// by a program to cause significant contention.)
	counters atomic.Pointer[Counter] // head of list
	end      Counter                 // list ends at &end instead of nil

	mu                 sync.Mutex
	buildInfo          *debug.BuildInfo
	timeBegin, timeEnd time.Time
	err                error
	current            atomic.Pointer[mappedFile] // may be read without holding mu, but may be nil
}

var defaultFile file

// register ensures that the counter c is registered with the file.
func (f *file) register(c *Counter) {
	debugPrintf("register %s %p\n", c.Name(), c)

	// If counter is not registered with file, register it.
	// Doing this lazily avoids init-time work
	// as well as any execution cost at all for counters
	// that are not used in a given program.
	wroteNext := false
	for wroteNext || c.next.Load() == nil {
		head := f.counters.Load()
		next := head
		if next == nil {
			next = &f.end
		}
		debugPrintf("register %s next %p\n", c.Name(), next)
		if !wroteNext {
			if !c.next.CompareAndSwap(nil, next) {
				debugPrintf("register %s cas failed %p\n", c.Name(), c.next.Load())
				continue
			}
			wroteNext = true
		} else {
			c.next.Store(next)
		}
		if f.counters.CompareAndSwap(head, c) {
			debugPrintf("registered %s %p\n", c.Name(), f.counters.Load())
			return
		}
		debugPrintf("register %s cas2 failed %p %p\n", c.Name(), f.counters.Load(), head)
	}
}

// invalidateCounters marks as invalid all the pointers
// held by f's counters and then refreshes them.
//
// invalidateCounters cannot be called while holding f.mu,
// because a counter refresh may call f.lookup.
func (f *file) invalidateCounters() {
	// Mark every counter as needing to refresh its count pointer.
	if head := f.counters.Load(); head != nil {
		for c := head; c != &f.end; c = c.next.Load() {
			c.invalidate()
		}
		for c := head; c != &f.end; c = c.next.Load() {
			c.refresh()
		}
	}
}

// lookup looks up the counter with the given name in the file,
// allocating it if needed, and returns a pointer to the atomic.Uint64
// containing the counter data.
// If the file has not been opened yet, lookup returns nil.
func (f *file) lookup(name string) counterPtr {
	current := f.current.Load()
	if current == nil {
		debugPrintf("lookup %s - no mapped file\n", name)
		return counterPtr{}
	}
	ptr := f.newCounter(name)
	if ptr == nil {
		return counterPtr{}
	}
	return counterPtr{current, ptr}
}

// ErrDisabled is the error returned when telemetry is disabled.
var ErrDisabled = errors.New("counter: disabled as Go telemetry is off")

var (
	errNoBuildInfo = errors.New("counter: missing build info")
	errCorrupt     = errors.New("counter: corrupt counter file")
)

// weekEnd returns the day of the week on which uploads occur (and therefore
// counters expire).
//
// Reads the weekends file, creating one if none exists.
func weekEnd() (time.Weekday, error) {
	// If there is no 'weekends' file create it and initialize it
	// to a random day of the week. There is a short interval for
	// a race.
	weekends := filepath.Join(telemetry.Default.LocalDir(), "weekends")
	day := fmt.Sprintf("%d\n", rand.Intn(7))
	if _, err := os.ReadFile(weekends); err != nil {
		if err := os.MkdirAll(telemetry.Default.LocalDir(), 0777); err != nil {
			debugPrintf("%v: could not create telemetry.LocalDir %s", err, telemetry.Default.LocalDir())
			return 0, err
		}
		if err = os.WriteFile(weekends, []byte(day), 0666); err != nil {
			return 0, err
		}
	}

	// race is over, read the file
	buf, err := os.ReadFile(weekends)
	// There is no reasonable way of recovering from errors
	// so we just fail
	if err != nil {
		return 0, err
	}
	buf = bytes.TrimSpace(buf)
	if len(buf) == 0 {
		return 0, fmt.Errorf("empty weekends file")
	}
	weekend := time.Weekday(buf[0] - '0') // 0 is Sunday
	// paranoia to make sure the value is legal
	weekend %= 7
	if weekend < 0 {
		weekend += 7
	}
	return weekend, nil
}

// rotate checks to see whether the file f needs to be rotated,
// meaning to start a new counter file with a different date in the name.
// rotate is also used to open the file initially, meaning f.current can be nil.
// In general rotate should be called just once for each file.
// rotate will arrange a timer to call itself again when necessary.
func (f *file) rotate() {
	expiry := f.rotate1()
	if !expiry.IsZero() {
		delay := time.Until(expiry)
		// Some tests set CounterTime to a time in the past, causing delay to be
		// negative. Avoid infinite loops by delaying at least a short interval.
		//
		// TODO(rfindley): instead, just also mock AfterFunc.
		const minDelay = 1 * time.Minute
		if delay < minDelay {
			delay = minDelay
		}
		// TODO(rsc): Does this do the right thing for laptops closing?
		time.AfterFunc(delay, f.rotate)
	}
}

func nop() {}

// CounterTime returns the current UTC time.
// Mutable for testing.
var CounterTime = func() time.Time {
	return time.Now().UTC()
}

// counterSpan returns the current time span for a counter file, as determined
// by [CounterTime] and the [weekEnd].
func counterSpan() (begin, end time.Time, _ error) {
	year, month, day := CounterTime().Date()
	begin = time.Date(year, month, day, 0, 0, 0, 0, time.UTC)
	// files always begin today, but expire on the next day of the week
	// from the 'weekends' file.
	weekend, err := weekEnd()
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	incr := int(weekend - begin.Weekday())
	if incr <= 0 {
		incr += 7 // ensure that end is later than begin
	}
	end = time.Date(year, month, day+incr, 0, 0, 0, 0, time.UTC)
	return begin, end, nil
}

// rotate1 rotates the current counter file, returning its expiry, or the zero
// time if rotation failed.
func (f *file) rotate1() time.Time {
	// Cleanup must be performed while unlocked, since invalidateCounters may
	// involve calls to f.lookup.
	var previous *mappedFile // read below while holding the f.mu.
	defer func() {
		// Counters must be invalidated whenever the mapped file changes.
		if next := f.current.Load(); next != previous {
			f.invalidateCounters()
			// Ensure that the previous counter mapped file is closed.
			if previous != nil {
				previous.close() // safe to call multiple times
			}
		}
	}()

	f.mu.Lock()
	defer f.mu.Unlock()

	previous = f.current.Load()

	if f.err != nil {
		return time.Time{} // already in failed state; nothing to do
	}

	fail := func(err error) {
		debugPrintf("rotate: %v", err)
		f.err = err
		f.current.Store(nil)
	}

	if mode, _ := telemetry.Default.Mode(); mode == "off" {
		// TODO(rfindley): do we ever want to make ErrDisabled recoverable?
		// Specifically, if f.err is ErrDisabled, should we check again during when
		// rotating?
		fail(ErrDisabled)
		return time.Time{}
	}

	if f.buildInfo == nil {
		bi, ok := debug.ReadBuildInfo()
		if !ok {
			fail(errNoBuildInfo)
			return time.Time{}
		}
		f.buildInfo = bi
	}

	begin, end, err := counterSpan()
	if err != nil {
		fail(err)
		return time.Time{}
	}
	if f.timeBegin.Equal(begin) && f.timeEnd.Equal(end) {
		return f.timeEnd // nothing to do
	}
	f.timeBegin, f.timeEnd = begin, end

	goVers, progPath, progVers := telemetry.ProgramInfo(f.buildInfo)
	meta := fmt.Sprintf("TimeBegin: %s\nTimeEnd: %s\nProgram: %s\nVersion: %s\nGoVersion: %s\nGOOS: %s\nGOARCH: %s\n\n",
		f.timeBegin.Format(time.RFC3339), f.timeEnd.Format(time.RFC3339),
		progPath, progVers, goVers, runtime.GOOS, runtime.GOARCH)
	if len(meta) > maxMetaLen { // should be impossible for our use
		fail(fmt.Errorf("metadata too long"))
		return time.Time{}
	}

	if progVers != "" {
		progVers = "@" + progVers
	}
	baseName := fmt.Sprintf("%s%s-%s-%s-%s-%s.%s.count",
		path.Base(progPath),
		progVers,
		goVers,
		runtime.GOOS,
		runtime.GOARCH,
		f.timeBegin.Format("2006-01-02"),
		FileVersion,
	)
	dir := telemetry.Default.LocalDir()
	if err := os.MkdirAll(dir, 0777); err != nil {
		fail(fmt.Errorf("making local dir: %v", err))
		return time.Time{}
	}
	name := filepath.Join(dir, baseName)

	m, err := openMapped(name, meta, nil)
	if err != nil {
		// Mapping failed:
		// If there used to be a mapped file, after cleanup
		// incrementing counters will only change their internal state.
		// (before cleanup the existing mapped file would be updated)
		fail(fmt.Errorf("openMapped: %v", err))
		return time.Time{}
	}

	debugPrintf("using %v", m.f.Name())
	f.current.Store(m)
	return f.timeEnd
}

func (f *file) newCounter(name string) *atomic.Uint64 {
	v, cleanup := f.newCounter1(name)
	cleanup()
	return v
}

func (f *file) newCounter1(name string) (v *atomic.Uint64, cleanup func()) {
	f.mu.Lock()
	defer f.mu.Unlock()

	current := f.current.Load()
	if current == nil {
		return nil, nop
	}
	debugPrintf("newCounter %s in %s\n", name, current.f.Name())
	if v, _, _, _ := current.lookup(name); v != nil {
		return v, nop
	}
	v, newM, err := current.newCounter(name)
	if err != nil {
		debugPrintf("newCounter %s: %v\n", name, err)
		return nil, nop
	}

	cleanup = nop
	if newM != nil {
		f.current.Store(newM)
		// TODO(rfindley): shouldn't this close f.current?
		cleanup = f.invalidateCounters
	}
	return v, cleanup
}

var openOnce sync.Once

// Open associates counting with the defaultFile.
// The returned function is for testing only, and should
// be called after all Inc()s are finished, but before
// any reports are generated.
// (Otherwise expired count files will not be deleted on Windows.)
func Open() func() {
	if telemetry.DisabledOnPlatform {
		return func() {}
	}
	close := func() {}
	openOnce.Do(func() {
		if mode, _ := telemetry.Default.Mode(); mode == "off" {
			// Don't open the file when telemetry is off.
			defaultFile.err = ErrDisabled
			// No need to clean up.
			return
		}
		debugPrintf("Open")
		defaultFile.rotate()
		close = func() {
			// Once this has been called, the defaultFile is no longer usable.
			mf := defaultFile.current.Load()
			if mf == nil {
				// telemetry might have been off
				return
			}
			mf.close()
		}
	})
	return close
}

// A mappedFile is a counter file mmapped into memory.
type mappedFile struct {
	meta      string
	hdrLen    uint32
	zero      [4]byte
	closeOnce sync.Once
	f         *os.File
	mapping   *mmap.Data
}

// existing should be nil the first time this is called for a file,
// and when remapping, should be the previous mappedFile.
func openMapped(name string, meta string, existing *mappedFile) (_ *mappedFile, err error) {
	hdr, err := mappedHeader(meta)
	if err != nil {
		return nil, err
	}

	f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return nil, err
	}
	// Note: using local variable m here, not return value,
	// so that return nil, err does not set m = nil and break the code in the defer.
	m := &mappedFile{
		f:    f,
		meta: meta,
	}
	// without this files cannot be cleanedup on Windows (affects tests)
	runtime.SetFinalizer(m, (*mappedFile).close)
	defer func() {
		if err != nil {
			m.close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}

	// Establish file header and initial data area if not already present.
	if info.Size() < minFileLen {
		if _, err := f.WriteAt(hdr, 0); err != nil {
			return nil, err
		}
		// Write zeros at the end of the file to extend it to minFileLen.
		if _, err := f.WriteAt(m.zero[:], int64(minFileLen-len(m.zero))); err != nil {
			return nil, err
		}
		info, err = f.Stat()
		if err != nil {
			return nil, err
		}
		if info.Size() < minFileLen {
			return nil, fmt.Errorf("counter: writing file did not extend it")
		}
	}

	// Map into memory.
	var mapping mmap.Data
	if existing != nil {
		mapping, err = memmap(f, existing.mapping)
	} else {
		mapping, err = memmap(f, nil)
	}
	if err != nil {
		return nil, err
	}
	m.mapping = &mapping
	if !bytes.HasPrefix(m.mapping.Data, hdr) {
		return nil, fmt.Errorf("counter: header mismatch")
	}
	m.hdrLen = uint32(len(hdr))

	return m, nil
}

const (
	FileVersion = "v1"
	hdrPrefix   = "# telemetry/counter file " + FileVersion + "\n"
	recordUnit  = 32
	maxMetaLen  = 512
	numHash     = 512 // 2kB for hash table
	maxNameLen  = 4 * 1024
	limitOff    = 0
	hashOff     = 4
	pageSize    = 16 * 1024
	minFileLen  = 16 * 1024
)

func mappedHeader(meta string) ([]byte, error) {
	if len(meta) > maxMetaLen {
		return nil, fmt.Errorf("counter: metadata too large")
	}
	np := round(len(hdrPrefix), 4)
	n := round(np+4+len(meta), 32)
	hdr := make([]byte, n)
	copy(hdr, hdrPrefix)
	*(*uint32)(unsafe.Pointer(&hdr[np])) = uint32(n)
	copy(hdr[np+4:], meta)
	return hdr, nil
}

func (m *mappedFile) place(limit uint32, name string) (start, end uint32) {
	if limit == 0 {
		// first record in file
		limit = m.hdrLen + hashOff + 4*numHash
	}
	n := round(uint32(16+len(name)), recordUnit)
	start = round(limit, recordUnit) // should already be rounded but just in case
	if start/pageSize != (start+n)/pageSize {
		// bump start to next page
		start = round(limit, pageSize)
	}
	return start, start + n
}

var memmap = mmap.Mmap
var munmap = mmap.Munmap

func (m *mappedFile) close() {
	m.closeOnce.Do(func() {
		if m.mapping != nil {
			munmap(m.mapping)
			m.mapping = nil
		}
		if m.f != nil {
			m.f.Close() // best effort
			m.f = nil
		}
	})
}

// hash returns the hash code for name.
// The implementation is FNV-1a.
// This hash function is a fixed detail of the file format.
// It cannot be changed without also changing the file format version.
func hash(name string) uint32 {
	const (
		offset32 = 2166136261
		prime32  = 16777619
	)
	h := uint32(offset32)
	for i := 0; i < len(name); i++ {
		c := name[i]
		h = (h ^ uint32(c)) * prime32
	}
	return (h ^ (h >> 16)) % numHash
}

func (m *mappedFile) load32(off uint32) uint32 {
	if int64(off) >= int64(len(m.mapping.Data)) {
		return 0
	}
	return (*atomic.Uint32)(unsafe.Pointer(&m.mapping.Data[off])).Load()
}

func (m *mappedFile) cas32(off, old, new uint32) bool {
	if int64(off) >= int64(len(m.mapping.Data)) {
		panic("bad cas32") // return false would probably loop
	}
	return (*atomic.Uint32)(unsafe.Pointer(&m.mapping.Data[off])).CompareAndSwap(old, new)
}

func (m *mappedFile) entryAt(off uint32) (name []byte, next uint32, v *atomic.Uint64, ok bool) {
	if off < m.hdrLen+hashOff || int64(off)+16 > int64(len(m.mapping.Data)) {
		return nil, 0, nil, false
	}
	nameLen := m.load32(off+8) & 0x00ffffff
	if nameLen == 0 || int64(off)+16+int64(nameLen) > int64(len(m.mapping.Data)) {
		return nil, 0, nil, false
	}
	name = m.mapping.Data[off+16 : off+16+nameLen]
	next = m.load32(off + 12)
	v = (*atomic.Uint64)(unsafe.Pointer(&m.mapping.Data[off]))
	return name, next, v, true
}

func (m *mappedFile) writeEntryAt(off uint32, name string) (next *atomic.Uint32, v *atomic.Uint64, ok bool) {
	if off < m.hdrLen+hashOff || int64(off)+16+int64(len(name)) > int64(len(m.mapping.Data)) {
		return nil, nil, false
	}
	copy(m.mapping.Data[off+16:], name)
	atomic.StoreUint32((*uint32)(unsafe.Pointer(&m.mapping.Data[off+8])), uint32(len(name))|0xff000000)
	next = (*atomic.Uint32)(unsafe.Pointer(&m.mapping.Data[off+12]))
	v = (*atomic.Uint64)(unsafe.Pointer(&m.mapping.Data[off]))
	return next, v, true
}

func (m *mappedFile) lookup(name string) (v *atomic.Uint64, headOff, head uint32, ok bool) {
	h := hash(name)
	headOff = m.hdrLen + hashOff + h*4
	head = m.load32(headOff)
	off := head
	for off != 0 {
		ename, next, v, ok := m.entryAt(off)
		if !ok {
			return nil, 0, 0, false
		}
		if string(ename) == name {
			return v, headOff, head, true
		}
		off = next
	}
	return nil, headOff, head, true
}

func (m *mappedFile) newCounter(name string) (v *atomic.Uint64, m1 *mappedFile, err error) {
	if len(name) > maxNameLen {
		return nil, nil, fmt.Errorf("counter name too long")
	}
	orig := m
	defer func() {
		if m != orig {
			if err != nil {
				m.close()
			} else {
				m1 = m
			}
		}
	}()

	v, headOff, head, ok := m.lookup(name)
	for !ok {
		// Lookup found an invalid pointer,
		// perhaps because the file has grown larger than the mapping.
		limit := m.load32(m.hdrLen + limitOff)
		if int64(limit) <= int64(len(m.mapping.Data)) {
			// Mapping doesn't need to grow, so lookup found actual corruption.
			debugPrintf("corrupt1\n")
			return nil, nil, errCorrupt
		}
		newM, err := openMapped(m.f.Name(), m.meta, m)
		if err != nil {
			return nil, nil, err
		}
		if m != orig {
			m.close()
		}
		m = newM
		v, headOff, head, ok = m.lookup(name)
	}
	if v != nil {
		return v, nil, nil
	}

	// Reserve space for new record.
	// We are competing against other programs using the same file,
	// so we use a compare-and-swap on the allocation limit in the header.
	var start, end uint32
	for {
		// Determine where record should end, and grow file if needed.
		limit := m.load32(m.hdrLen + limitOff)
		start, end = m.place(limit, name)
		debugPrintf("place %s at %#x-%#x\n", name, start, end)
		if int64(end) > int64(len(m.mapping.Data)) {
			newM, err := m.extend(end)
			if err != nil {
				return nil, nil, err
			}
			if m != orig {
				m.close()
			}
			m = newM
			continue
		}

		// Attempt to reserve that space for our record.
		if m.cas32(m.hdrLen+limitOff, limit, end) {
			break
		}
	}

	// Write record.
	next, v, ok := m.writeEntryAt(start, name)
	if !ok {
		debugPrintf("corrupt2 %#x+%d vs %#x\n", start, len(name), len(m.mapping.Data))
		return nil, nil, errCorrupt // more likely our math is wrong
	}

	// Link record into hash chain, making sure not to introduce a duplicate.
	// We know name does not appear in the chain starting at head.
	for {
		next.Store(head)
		if m.cas32(headOff, head, start) {
			return v, nil, nil
		}

		// Check new elements in chain for duplicates.
		old := head
		head = m.load32(headOff)
		for off := head; off != old; {
			ename, enext, v, ok := m.entryAt(off)
			if !ok {
				return nil, nil, errCorrupt
			}
			if string(ename) == name {
				next.Store(^uint32(0)) // mark ours as dead
				return v, nil, nil
			}
			off = enext
		}
	}
}

func (m *mappedFile) extend(end uint32) (*mappedFile, error) {
	end = round(end, pageSize)
	info, err := m.f.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < int64(end) {
		if _, err := m.f.WriteAt(m.zero[:], int64(end)-int64(len(m.zero))); err != nil {
			return nil, err
		}
	}
	newM, err := openMapped(m.f.Name(), m.meta, m)
	m.f.Close()
	return newM, err
}

// round returns x rounded up to the next multiple of unit,
// which must be a power of two.
func round[T int | uint32](x T, unit T) T {
	return (x + unit - 1) &^ (unit - 1)
}
