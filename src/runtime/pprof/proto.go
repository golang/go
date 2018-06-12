// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"runtime"
	"strconv"
	"time"
	"unsafe"
)

// lostProfileEvent is the function to which lost profiling
// events are attributed.
// (The name shows up in the pprof graphs.)
func lostProfileEvent() { lostProfileEvent() }

// funcPC returns the PC for the func value f.
func funcPC(f interface{}) uintptr {
	return *(*[2]*uintptr)(unsafe.Pointer(&f))[1]
}

// A profileBuilder writes a profile incrementally from a
// stream of profile samples delivered by the runtime.
type profileBuilder struct {
	start      time.Time
	end        time.Time
	havePeriod bool
	period     int64
	m          profMap

	// encoding state
	w         io.Writer
	zw        *gzip.Writer
	pb        protobuf
	strings   []string
	stringMap map[string]int
	locs      map[uintptr]int
	funcs     map[string]int // Package path-qualified function name to Function.ID
	mem       []memMap
}

type memMap struct {
	// initialized as reading mapping
	start         uintptr
	end           uintptr
	offset        uint64
	file, buildID string

	funcs symbolizeFlag
}

// symbolizeFlag keeps track of symbolization result.
//   0                  : no symbol lookup was performed
//   1<<0 (lookupTried) : symbol lookup was performed
//   1<<1 (lookupFailed): symbol lookup was performed but failed
type symbolizeFlag uint8

const (
	lookupTried  symbolizeFlag = 1 << iota
	lookupFailed symbolizeFlag = 1 << iota
)

const (
	// message Profile
	tagProfile_SampleType        = 1  // repeated ValueType
	tagProfile_Sample            = 2  // repeated Sample
	tagProfile_Mapping           = 3  // repeated Mapping
	tagProfile_Location          = 4  // repeated Location
	tagProfile_Function          = 5  // repeated Function
	tagProfile_StringTable       = 6  // repeated string
	tagProfile_DropFrames        = 7  // int64 (string table index)
	tagProfile_KeepFrames        = 8  // int64 (string table index)
	tagProfile_TimeNanos         = 9  // int64
	tagProfile_DurationNanos     = 10 // int64
	tagProfile_PeriodType        = 11 // ValueType (really optional string???)
	tagProfile_Period            = 12 // int64
	tagProfile_Comment           = 13 // repeated int64
	tagProfile_DefaultSampleType = 14 // int64

	// message ValueType
	tagValueType_Type = 1 // int64 (string table index)
	tagValueType_Unit = 2 // int64 (string table index)

	// message Sample
	tagSample_Location = 1 // repeated uint64
	tagSample_Value    = 2 // repeated int64
	tagSample_Label    = 3 // repeated Label

	// message Label
	tagLabel_Key = 1 // int64 (string table index)
	tagLabel_Str = 2 // int64 (string table index)
	tagLabel_Num = 3 // int64

	// message Mapping
	tagMapping_ID              = 1  // uint64
	tagMapping_Start           = 2  // uint64
	tagMapping_Limit           = 3  // uint64
	tagMapping_Offset          = 4  // uint64
	tagMapping_Filename        = 5  // int64 (string table index)
	tagMapping_BuildID         = 6  // int64 (string table index)
	tagMapping_HasFunctions    = 7  // bool
	tagMapping_HasFilenames    = 8  // bool
	tagMapping_HasLineNumbers  = 9  // bool
	tagMapping_HasInlineFrames = 10 // bool

	// message Location
	tagLocation_ID        = 1 // uint64
	tagLocation_MappingID = 2 // uint64
	tagLocation_Address   = 3 // uint64
	tagLocation_Line      = 4 // repeated Line

	// message Line
	tagLine_FunctionID = 1 // uint64
	tagLine_Line       = 2 // int64

	// message Function
	tagFunction_ID         = 1 // uint64
	tagFunction_Name       = 2 // int64 (string table index)
	tagFunction_SystemName = 3 // int64 (string table index)
	tagFunction_Filename   = 4 // int64 (string table index)
	tagFunction_StartLine  = 5 // int64
)

// stringIndex adds s to the string table if not already present
// and returns the index of s in the string table.
func (b *profileBuilder) stringIndex(s string) int64 {
	id, ok := b.stringMap[s]
	if !ok {
		id = len(b.strings)
		b.strings = append(b.strings, s)
		b.stringMap[s] = id
	}
	return int64(id)
}

func (b *profileBuilder) flush() {
	const dataFlush = 4096
	if b.pb.nest == 0 && len(b.pb.data) > dataFlush {
		b.zw.Write(b.pb.data)
		b.pb.data = b.pb.data[:0]
	}
}

// pbValueType encodes a ValueType message to b.pb.
func (b *profileBuilder) pbValueType(tag int, typ, unit string) {
	start := b.pb.startMessage()
	b.pb.int64(tagValueType_Type, b.stringIndex(typ))
	b.pb.int64(tagValueType_Unit, b.stringIndex(unit))
	b.pb.endMessage(tag, start)
}

// pbSample encodes a Sample message to b.pb.
func (b *profileBuilder) pbSample(values []int64, locs []uint64, labels func()) {
	start := b.pb.startMessage()
	b.pb.int64s(tagSample_Value, values)
	b.pb.uint64s(tagSample_Location, locs)
	if labels != nil {
		labels()
	}
	b.pb.endMessage(tagProfile_Sample, start)
	b.flush()
}

// pbLabel encodes a Label message to b.pb.
func (b *profileBuilder) pbLabel(tag int, key, str string, num int64) {
	start := b.pb.startMessage()
	b.pb.int64Opt(tagLabel_Key, b.stringIndex(key))
	b.pb.int64Opt(tagLabel_Str, b.stringIndex(str))
	b.pb.int64Opt(tagLabel_Num, num)
	b.pb.endMessage(tag, start)
}

// pbLine encodes a Line message to b.pb.
func (b *profileBuilder) pbLine(tag int, funcID uint64, line int64) {
	start := b.pb.startMessage()
	b.pb.uint64Opt(tagLine_FunctionID, funcID)
	b.pb.int64Opt(tagLine_Line, line)
	b.pb.endMessage(tag, start)
}

// pbMapping encodes a Mapping message to b.pb.
func (b *profileBuilder) pbMapping(tag int, id, base, limit, offset uint64, file, buildID string, hasFuncs bool) {
	start := b.pb.startMessage()
	b.pb.uint64Opt(tagMapping_ID, id)
	b.pb.uint64Opt(tagMapping_Start, base)
	b.pb.uint64Opt(tagMapping_Limit, limit)
	b.pb.uint64Opt(tagMapping_Offset, offset)
	b.pb.int64Opt(tagMapping_Filename, b.stringIndex(file))
	b.pb.int64Opt(tagMapping_BuildID, b.stringIndex(buildID))
	// TODO: we set HasFunctions if all symbols from samples were symbolized (hasFuncs).
	// Decide what to do about HasInlineFrames and HasLineNumbers.
	// Also, another approach to handle the mapping entry with
	// incomplete symbolization results is to dupliace the mapping
	// entry (but with different Has* fields values) and use
	// different entries for symbolized locations and unsymbolized locations.
	if hasFuncs {
		b.pb.bool(tagMapping_HasFunctions, true)
	}
	b.pb.endMessage(tag, start)
}

// locForPC returns the location ID for addr.
// addr must be a return PC. This returns the location of the call.
// It may emit to b.pb, so there must be no message encoding in progress.
func (b *profileBuilder) locForPC(addr uintptr) uint64 {
	id := uint64(b.locs[addr])
	if id != 0 {
		return id
	}

	// Expand this one address using CallersFrames so we can cache
	// each expansion. In general, CallersFrames takes a whole
	// stack, but in this case we know there will be no skips in
	// the stack and we have return PCs anyway.
	frames := runtime.CallersFrames([]uintptr{addr})
	frame, more := frames.Next()
	if frame.Function == "runtime.goexit" {
		// Short-circuit if we see runtime.goexit so the loop
		// below doesn't allocate a useless empty location.
		return 0
	}

	symbolizeResult := lookupTried
	if frame.PC == 0 || frame.Function == "" || frame.File == "" || frame.Line == 0 {
		symbolizeResult |= lookupFailed
	}

	if frame.PC == 0 {
		// If we failed to resolve the frame, at least make up
		// a reasonable call PC. This mostly happens in tests.
		frame.PC = addr - 1
	}

	// We can't write out functions while in the middle of the
	// Location message, so record new functions we encounter and
	// write them out after the Location.
	type newFunc struct {
		id         uint64
		name, file string
	}
	newFuncs := make([]newFunc, 0, 8)

	id = uint64(len(b.locs)) + 1
	b.locs[addr] = int(id)
	start := b.pb.startMessage()
	b.pb.uint64Opt(tagLocation_ID, id)
	b.pb.uint64Opt(tagLocation_Address, uint64(frame.PC))
	for frame.Function != "runtime.goexit" {
		// Write out each line in frame expansion.
		funcID := uint64(b.funcs[frame.Function])
		if funcID == 0 {
			funcID = uint64(len(b.funcs)) + 1
			b.funcs[frame.Function] = int(funcID)
			newFuncs = append(newFuncs, newFunc{funcID, frame.Function, frame.File})
		}
		b.pbLine(tagLocation_Line, funcID, int64(frame.Line))
		if !more {
			break
		}
		frame, more = frames.Next()
	}
	for i := range b.mem {
		if b.mem[i].start <= addr && addr < b.mem[i].end {
			b.pb.uint64Opt(tagLocation_MappingID, uint64(i+1))

			m := b.mem[i]
			m.funcs |= symbolizeResult
			b.mem[i] = m
			break
		}
	}
	b.pb.endMessage(tagProfile_Location, start)

	// Write out functions we found during frame expansion.
	for _, fn := range newFuncs {
		start := b.pb.startMessage()
		b.pb.uint64Opt(tagFunction_ID, fn.id)
		b.pb.int64Opt(tagFunction_Name, b.stringIndex(fn.name))
		b.pb.int64Opt(tagFunction_SystemName, b.stringIndex(fn.name))
		b.pb.int64Opt(tagFunction_Filename, b.stringIndex(fn.file))
		b.pb.endMessage(tagProfile_Function, start)
	}

	b.flush()
	return id
}

// newProfileBuilder returns a new profileBuilder.
// CPU profiling data obtained from the runtime can be added
// by calling b.addCPUData, and then the eventual profile
// can be obtained by calling b.finish.
func newProfileBuilder(w io.Writer) *profileBuilder {
	zw, _ := gzip.NewWriterLevel(w, gzip.BestSpeed)
	b := &profileBuilder{
		w:         w,
		zw:        zw,
		start:     time.Now(),
		strings:   []string{""},
		stringMap: map[string]int{"": 0},
		locs:      map[uintptr]int{},
		funcs:     map[string]int{},
	}
	b.readMapping()
	return b
}

// addCPUData adds the CPU profiling data to the profile.
// The data must be a whole number of records,
// as delivered by the runtime.
func (b *profileBuilder) addCPUData(data []uint64, tags []unsafe.Pointer) error {
	if !b.havePeriod {
		// first record is period
		if len(data) < 3 {
			return fmt.Errorf("truncated profile")
		}
		if data[0] != 3 || data[2] == 0 {
			return fmt.Errorf("malformed profile")
		}
		// data[2] is sampling rate in Hz. Convert to sampling
		// period in nanoseconds.
		b.period = 1e9 / int64(data[2])
		b.havePeriod = true
		data = data[3:]
	}

	// Parse CPU samples from the profile.
	// Each sample is 3+n uint64s:
	//	data[0] = 3+n
	//	data[1] = time stamp (ignored)
	//	data[2] = count
	//	data[3:3+n] = stack
	// If the count is 0 and the stack has length 1,
	// that's an overflow record inserted by the runtime
	// to indicate that stack[0] samples were lost.
	// Otherwise the count is usually 1,
	// but in a few special cases like lost non-Go samples
	// there can be larger counts.
	// Because many samples with the same stack arrive,
	// we want to deduplicate immediately, which we do
	// using the b.m profMap.
	for len(data) > 0 {
		if len(data) < 3 || data[0] > uint64(len(data)) {
			return fmt.Errorf("truncated profile")
		}
		if data[0] < 3 || tags != nil && len(tags) < 1 {
			return fmt.Errorf("malformed profile")
		}
		count := data[2]
		stk := data[3:data[0]]
		data = data[data[0]:]
		var tag unsafe.Pointer
		if tags != nil {
			tag = tags[0]
			tags = tags[1:]
		}

		if count == 0 && len(stk) == 1 {
			// overflow record
			count = uint64(stk[0])
			stk = []uint64{
				uint64(funcPC(lostProfileEvent)),
			}
		}
		b.m.lookup(stk, tag).count += int64(count)
	}
	return nil
}

// build completes and returns the constructed profile.
func (b *profileBuilder) build() {
	b.end = time.Now()

	b.pb.int64Opt(tagProfile_TimeNanos, b.start.UnixNano())
	if b.havePeriod { // must be CPU profile
		b.pbValueType(tagProfile_SampleType, "samples", "count")
		b.pbValueType(tagProfile_SampleType, "cpu", "nanoseconds")
		b.pb.int64Opt(tagProfile_DurationNanos, b.end.Sub(b.start).Nanoseconds())
		b.pbValueType(tagProfile_PeriodType, "cpu", "nanoseconds")
		b.pb.int64Opt(tagProfile_Period, b.period)
	}

	values := []int64{0, 0}
	var locs []uint64
	for e := b.m.all; e != nil; e = e.nextAll {
		values[0] = e.count
		values[1] = e.count * b.period

		var labels func()
		if e.tag != nil {
			labels = func() {
				for k, v := range *(*labelMap)(e.tag) {
					b.pbLabel(tagSample_Label, k, v, 0)
				}
			}
		}

		locs = locs[:0]
		for i, addr := range e.stk {
			// Addresses from stack traces point to the
			// next instruction after each call, except
			// for the leaf, which points to where the
			// signal occurred. locForPC expects return
			// PCs, so increment the leaf address to look
			// like a return PC.
			if i == 0 {
				addr++
			}
			l := b.locForPC(addr)
			if l == 0 { // runtime.goexit
				continue
			}
			locs = append(locs, l)
		}
		b.pbSample(values, locs, labels)
	}

	for i, m := range b.mem {
		hasFunctions := m.funcs == lookupTried // lookupTried but not lookupFailed
		b.pbMapping(tagProfile_Mapping, uint64(i+1), uint64(m.start), uint64(m.end), m.offset, m.file, m.buildID, hasFunctions)
	}

	// TODO: Anything for tagProfile_DropFrames?
	// TODO: Anything for tagProfile_KeepFrames?

	b.pb.strings(tagProfile_StringTable, b.strings)
	b.zw.Write(b.pb.data)
	b.zw.Close()
}

// readMapping reads /proc/self/maps and writes mappings to b.pb.
// It saves the address ranges of the mappings in b.mem for use
// when emitting locations.
func (b *profileBuilder) readMapping() {
	data, _ := ioutil.ReadFile("/proc/self/maps")
	parseProcSelfMaps(data, b.addMapping)
}

func parseProcSelfMaps(data []byte, addMapping func(lo, hi, offset uint64, file, buildID string)) {
	// $ cat /proc/self/maps
	// 00400000-0040b000 r-xp 00000000 fc:01 787766                             /bin/cat
	// 0060a000-0060b000 r--p 0000a000 fc:01 787766                             /bin/cat
	// 0060b000-0060c000 rw-p 0000b000 fc:01 787766                             /bin/cat
	// 014ab000-014cc000 rw-p 00000000 00:00 0                                  [heap]
	// 7f7d76af8000-7f7d7797c000 r--p 00000000 fc:01 1318064                    /usr/lib/locale/locale-archive
	// 7f7d7797c000-7f7d77b36000 r-xp 00000000 fc:01 1180226                    /lib/x86_64-linux-gnu/libc-2.19.so
	// 7f7d77b36000-7f7d77d36000 ---p 001ba000 fc:01 1180226                    /lib/x86_64-linux-gnu/libc-2.19.so
	// 7f7d77d36000-7f7d77d3a000 r--p 001ba000 fc:01 1180226                    /lib/x86_64-linux-gnu/libc-2.19.so
	// 7f7d77d3a000-7f7d77d3c000 rw-p 001be000 fc:01 1180226                    /lib/x86_64-linux-gnu/libc-2.19.so
	// 7f7d77d3c000-7f7d77d41000 rw-p 00000000 00:00 0
	// 7f7d77d41000-7f7d77d64000 r-xp 00000000 fc:01 1180217                    /lib/x86_64-linux-gnu/ld-2.19.so
	// 7f7d77f3f000-7f7d77f42000 rw-p 00000000 00:00 0
	// 7f7d77f61000-7f7d77f63000 rw-p 00000000 00:00 0
	// 7f7d77f63000-7f7d77f64000 r--p 00022000 fc:01 1180217                    /lib/x86_64-linux-gnu/ld-2.19.so
	// 7f7d77f64000-7f7d77f65000 rw-p 00023000 fc:01 1180217                    /lib/x86_64-linux-gnu/ld-2.19.so
	// 7f7d77f65000-7f7d77f66000 rw-p 00000000 00:00 0
	// 7ffc342a2000-7ffc342c3000 rw-p 00000000 00:00 0                          [stack]
	// 7ffc34343000-7ffc34345000 r-xp 00000000 00:00 0                          [vdso]
	// ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]

	var line []byte
	// next removes and returns the next field in the line.
	// It also removes from line any spaces following the field.
	next := func() []byte {
		j := bytes.IndexByte(line, ' ')
		if j < 0 {
			f := line
			line = nil
			return f
		}
		f := line[:j]
		line = line[j+1:]
		for len(line) > 0 && line[0] == ' ' {
			line = line[1:]
		}
		return f
	}

	for len(data) > 0 {
		i := bytes.IndexByte(data, '\n')
		if i < 0 {
			line, data = data, nil
		} else {
			line, data = data[:i], data[i+1:]
		}
		addr := next()
		i = bytes.IndexByte(addr, '-')
		if i < 0 {
			continue
		}
		lo, err := strconv.ParseUint(string(addr[:i]), 16, 64)
		if err != nil {
			continue
		}
		hi, err := strconv.ParseUint(string(addr[i+1:]), 16, 64)
		if err != nil {
			continue
		}
		perm := next()
		if len(perm) < 4 || perm[2] != 'x' {
			// Only interested in executable mappings.
			continue
		}
		offset, err := strconv.ParseUint(string(next()), 16, 64)
		if err != nil {
			continue
		}
		next()          // dev
		inode := next() // inode
		if line == nil {
			continue
		}
		file := string(line)
		if len(inode) == 1 && inode[0] == '0' && file == "" {
			// Huge-page text mappings list the initial fragment of
			// mapped but unpopulated memory as being inode 0.
			// Don't report that part.
			// But [vdso] and [vsyscall] are inode 0, so let non-empty file names through.
			continue
		}

		// TODO: pprof's remapMappingIDs makes two adjustments:
		// 1. If there is an /anon_hugepage mapping first and it is
		// consecutive to a next mapping, drop the /anon_hugepage.
		// 2. If start-offset = 0x400000, change start to 0x400000 and offset to 0.
		// There's no indication why either of these is needed.
		// Let's try not doing these and see what breaks.
		// If we do need them, they would go here, before we
		// enter the mappings into b.mem in the first place.

		buildID, _ := elfBuildID(file)
		addMapping(lo, hi, offset, file, buildID)
	}
}

func (b *profileBuilder) addMapping(lo, hi, offset uint64, file, buildID string) {
	b.mem = append(b.mem, memMap{
		start:   uintptr(lo),
		end:     uintptr(hi),
		offset:  offset,
		file:    file,
		buildID: buildID,
	})
}
