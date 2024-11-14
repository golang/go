// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"internal/abi"
	"io"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unsafe"
)

// lostProfileEvent is the function to which lost profiling
// events are attributed.
// (The name shows up in the pprof graphs.)
func lostProfileEvent() { lostProfileEvent() }

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
	locs      map[uintptr]locInfo // list of locInfo starting with the given PC.
	funcs     map[string]int      // Package path-qualified function name to Function.ID
	mem       []memMap
	deck      pcDeck
}

type memMap struct {
	// initialized as reading mapping
	start   uintptr // Address at which the binary (or DLL) is loaded into memory.
	end     uintptr // The limit of the address range occupied by this mapping.
	offset  uint64  // Offset in the binary that corresponds to the first mapped address.
	file    string  // The object this entry is loaded from.
	buildID string  // A string that uniquely identifies a particular program version with high probability.

	funcs symbolizeFlag
	fake  bool // map entry was faked; /proc/self/maps wasn't available
}

// symbolizeFlag keeps track of symbolization result.
//
//	0                  : no symbol lookup was performed
//	1<<0 (lookupTried) : symbol lookup was performed
//	1<<1 (lookupFailed): symbol lookup was performed but failed
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
	// incomplete symbolization results is to duplicate the mapping
	// entry (but with different Has* fields values) and use
	// different entries for symbolized locations and unsymbolized locations.
	if hasFuncs {
		b.pb.bool(tagMapping_HasFunctions, true)
	}
	b.pb.endMessage(tag, start)
}

func allFrames(addr uintptr) ([]runtime.Frame, symbolizeFlag) {
	// Expand this one address using CallersFrames so we can cache
	// each expansion. In general, CallersFrames takes a whole
	// stack, but in this case we know there will be no skips in
	// the stack and we have return PCs anyway.
	frames := runtime.CallersFrames([]uintptr{addr})
	frame, more := frames.Next()
	if frame.Function == "runtime.goexit" {
		// Short-circuit if we see runtime.goexit so the loop
		// below doesn't allocate a useless empty location.
		return nil, 0
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
	ret := []runtime.Frame{frame}
	for frame.Function != "runtime.goexit" && more {
		frame, more = frames.Next()
		ret = append(ret, frame)
	}
	return ret, symbolizeResult
}

type locInfo struct {
	// location id assigned by the profileBuilder
	id uint64

	// sequence of PCs, including the fake PCs returned by the traceback
	// to represent inlined functions
	// https://github.com/golang/go/blob/d6f2f833c93a41ec1c68e49804b8387a06b131c5/src/runtime/traceback.go#L347-L368
	pcs []uintptr

	// firstPCFrames and firstPCSymbolizeResult hold the results of the
	// allFrames call for the first (leaf-most) PC this locInfo represents
	firstPCFrames          []runtime.Frame
	firstPCSymbolizeResult symbolizeFlag
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
		locs:      map[uintptr]locInfo{},
		funcs:     map[string]int{},
	}
	b.readMapping()
	return b
}

// addCPUData adds the CPU profiling data to the profile.
//
// The data must be a whole number of records, as delivered by the runtime.
// len(tags) must be equal to the number of records in data.
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
		// Consume tag slot. Note that there isn't a meaningful tag
		// value for this record.
		tags = tags[1:]
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
		if len(tags) < 1 {
			return fmt.Errorf("mismatched profile records and tags")
		}
		count := data[2]
		stk := data[3:data[0]]
		data = data[data[0]:]
		tag := tags[0]
		tags = tags[1:]

		if count == 0 && len(stk) == 1 {
			// overflow record
			count = uint64(stk[0])
			stk = []uint64{
				// gentraceback guarantees that PCs in the
				// stack can be unconditionally decremented and
				// still be valid, so we must do the same.
				uint64(abi.FuncPCABIInternal(lostProfileEvent) + 1),
			}
		}
		b.m.lookup(stk, tag).count += int64(count)
	}

	if len(tags) != 0 {
		return fmt.Errorf("mismatched profile records and tags")
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

		locs = b.appendLocsForStack(locs[:0], e.stk)

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

// appendLocsForStack appends the location IDs for the given stack trace to the given
// location ID slice, locs. The addresses in the stack are return PCs or 1 + the PC of
// an inline marker as the runtime traceback function returns.
//
// It may return an empty slice even if locs is non-empty, for example if locs consists
// solely of runtime.goexit. We still count these empty stacks in profiles in order to
// get the right cumulative sample count.
//
// It may emit to b.pb, so there must be no message encoding in progress.
func (b *profileBuilder) appendLocsForStack(locs []uint64, stk []uintptr) (newLocs []uint64) {
	b.deck.reset()

	// The last frame might be truncated. Recover lost inline frames.
	stk = runtime_expandFinalInlineFrame(stk)

	for len(stk) > 0 {
		addr := stk[0]
		if l, ok := b.locs[addr]; ok {
			// When generating code for an inlined function, the compiler adds
			// NOP instructions to the outermost function as a placeholder for
			// each layer of inlining. When the runtime generates tracebacks for
			// stacks that include inlined functions, it uses the addresses of
			// those NOPs as "fake" PCs on the stack as if they were regular
			// function call sites. But if a profiling signal arrives while the
			// CPU is executing one of those NOPs, its PC will show up as a leaf
			// in the profile with its own Location entry. So, always check
			// whether addr is a "fake" PC in the context of the current call
			// stack by trying to add it to the inlining deck before assuming
			// that the deck is complete.
			if len(b.deck.pcs) > 0 {
				if added := b.deck.tryAdd(addr, l.firstPCFrames, l.firstPCSymbolizeResult); added {
					stk = stk[1:]
					continue
				}
			}

			// first record the location if there is any pending accumulated info.
			if id := b.emitLocation(); id > 0 {
				locs = append(locs, id)
			}

			// then, record the cached location.
			locs = append(locs, l.id)

			// Skip the matching pcs.
			//
			// Even if stk was truncated due to the stack depth
			// limit, expandFinalInlineFrame above has already
			// fixed the truncation, ensuring it is long enough.
			stk = stk[len(l.pcs):]
			continue
		}

		frames, symbolizeResult := allFrames(addr)
		if len(frames) == 0 { // runtime.goexit.
			if id := b.emitLocation(); id > 0 {
				locs = append(locs, id)
			}
			stk = stk[1:]
			continue
		}

		if added := b.deck.tryAdd(addr, frames, symbolizeResult); added {
			stk = stk[1:]
			continue
		}
		// add failed because this addr is not inlined with the
		// existing PCs in the deck. Flush the deck and retry handling
		// this pc.
		if id := b.emitLocation(); id > 0 {
			locs = append(locs, id)
		}

		// check cache again - previous emitLocation added a new entry
		if l, ok := b.locs[addr]; ok {
			locs = append(locs, l.id)
			stk = stk[len(l.pcs):] // skip the matching pcs.
		} else {
			b.deck.tryAdd(addr, frames, symbolizeResult) // must succeed.
			stk = stk[1:]
		}
	}
	if id := b.emitLocation(); id > 0 { // emit remaining location.
		locs = append(locs, id)
	}
	return locs
}

// Here's an example of how Go 1.17 writes out inlined functions, compiled for
// linux/amd64. The disassembly of main.main shows two levels of inlining: main
// calls b, b calls a, a does some work.
//
//   inline.go:9   0x4553ec  90              NOPL                 // func main()    { b(v) }
//   inline.go:6   0x4553ed  90              NOPL                 // func b(v *int) { a(v) }
//   inline.go:5   0x4553ee  48c7002a000000  MOVQ $0x2a, 0(AX)    // func a(v *int) { *v = 42 }
//
// If a profiling signal arrives while executing the MOVQ at 0x4553ee (for line
// 5), the runtime will report the stack as the MOVQ frame being called by the
// NOPL at 0x4553ed (for line 6) being called by the NOPL at 0x4553ec (for line
// 9).
//
// The role of pcDeck is to collapse those three frames back into a single
// location at 0x4553ee, with file/line/function symbolization info representing
// the three layers of calls. It does that via sequential calls to pcDeck.tryAdd
// starting with the leaf-most address. The fourth call to pcDeck.tryAdd will be
// for the caller of main.main. Because main.main was not inlined in its caller,
// the deck will reject the addition, and the fourth PC on the stack will get
// its own location.

// pcDeck is a helper to detect a sequence of inlined functions from
// a stack trace returned by the runtime.
//
// The stack traces returned by runtime's trackback functions are fully
// expanded (at least for Go functions) and include the fake pcs representing
// inlined functions. The profile proto expects the inlined functions to be
// encoded in one Location message.
// https://github.com/google/pprof/blob/5e965273ee43930341d897407202dd5e10e952cb/proto/profile.proto#L177-L184
//
// Runtime does not directly expose whether a frame is for an inlined function
// and looking up debug info is not ideal, so we use a heuristic to filter
// the fake pcs and restore the inlined and entry functions. Inlined functions
// have the following properties:
//
//	Frame's Func is nil (note: also true for non-Go functions), and
//	Frame's Entry matches its entry function frame's Entry (note: could also be true for recursive calls and non-Go functions), and
//	Frame's Name does not match its entry function frame's name (note: inlined functions cannot be directly recursive).
//
// As reading and processing the pcs in a stack trace one by one (from leaf to the root),
// we use pcDeck to temporarily hold the observed pcs and their expanded frames
// until we observe the entry function frame.
type pcDeck struct {
	pcs             []uintptr
	frames          []runtime.Frame
	symbolizeResult symbolizeFlag

	// firstPCFrames indicates the number of frames associated with the first
	// (leaf-most) PC in the deck
	firstPCFrames int
	// firstPCSymbolizeResult holds the results of the allFrames call for the
	// first (leaf-most) PC in the deck
	firstPCSymbolizeResult symbolizeFlag
}

func (d *pcDeck) reset() {
	d.pcs = d.pcs[:0]
	d.frames = d.frames[:0]
	d.symbolizeResult = 0
	d.firstPCFrames = 0
	d.firstPCSymbolizeResult = 0
}

// tryAdd tries to add the pc and Frames expanded from it (most likely one,
// since the stack trace is already fully expanded) and the symbolizeResult
// to the deck. If it fails the caller needs to flush the deck and retry.
func (d *pcDeck) tryAdd(pc uintptr, frames []runtime.Frame, symbolizeResult symbolizeFlag) (success bool) {
	if existing := len(d.frames); existing > 0 {
		// 'd.frames' are all expanded from one 'pc' and represent all
		// inlined functions so we check only the last one.
		newFrame := frames[0]
		last := d.frames[existing-1]
		if last.Func != nil { // the last frame can't be inlined. Flush.
			return false
		}
		if last.Entry == 0 || newFrame.Entry == 0 { // Possibly not a Go function. Don't try to merge.
			return false
		}

		if last.Entry != newFrame.Entry { // newFrame is for a different function.
			return false
		}
		if runtime_FrameSymbolName(&last) == runtime_FrameSymbolName(&newFrame) { // maybe recursion.
			return false
		}
	}
	d.pcs = append(d.pcs, pc)
	d.frames = append(d.frames, frames...)
	d.symbolizeResult |= symbolizeResult
	if len(d.pcs) == 1 {
		d.firstPCFrames = len(d.frames)
		d.firstPCSymbolizeResult = symbolizeResult
	}
	return true
}

// emitLocation emits the new location and function information recorded in the deck
// and returns the location ID encoded in the profile protobuf.
// It emits to b.pb, so there must be no message encoding in progress.
// It resets the deck.
func (b *profileBuilder) emitLocation() uint64 {
	if len(b.deck.pcs) == 0 {
		return 0
	}
	defer b.deck.reset()

	addr := b.deck.pcs[0]
	firstFrame := b.deck.frames[0]

	// We can't write out functions while in the middle of the
	// Location message, so record new functions we encounter and
	// write them out after the Location.
	type newFunc struct {
		id         uint64
		name, file string
		startLine  int64
	}
	newFuncs := make([]newFunc, 0, 8)

	id := uint64(len(b.locs)) + 1
	b.locs[addr] = locInfo{
		id:                     id,
		pcs:                    append([]uintptr{}, b.deck.pcs...),
		firstPCSymbolizeResult: b.deck.firstPCSymbolizeResult,
		firstPCFrames:          append([]runtime.Frame{}, b.deck.frames[:b.deck.firstPCFrames]...),
	}

	start := b.pb.startMessage()
	b.pb.uint64Opt(tagLocation_ID, id)
	b.pb.uint64Opt(tagLocation_Address, uint64(firstFrame.PC))
	for _, frame := range b.deck.frames {
		// Write out each line in frame expansion.
		funcName := runtime_FrameSymbolName(&frame)
		funcID := uint64(b.funcs[funcName])
		if funcID == 0 {
			funcID = uint64(len(b.funcs)) + 1
			b.funcs[funcName] = int(funcID)
			newFuncs = append(newFuncs, newFunc{
				id:        funcID,
				name:      funcName,
				file:      frame.File,
				startLine: int64(runtime_FrameStartLine(&frame)),
			})
		}
		b.pbLine(tagLocation_Line, funcID, int64(frame.Line))
	}
	for i := range b.mem {
		if b.mem[i].start <= addr && addr < b.mem[i].end || b.mem[i].fake {
			b.pb.uint64Opt(tagLocation_MappingID, uint64(i+1))

			m := b.mem[i]
			m.funcs |= b.deck.symbolizeResult
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
		b.pb.int64Opt(tagFunction_StartLine, fn.startLine)
		b.pb.endMessage(tagProfile_Function, start)
	}

	b.flush()
	return id
}

var space = []byte(" ")
var newline = []byte("\n")

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
	next := func {
		var f []byte
		f, line, _ = bytes.Cut(line, space)
		line = bytes.TrimLeft(line, " ")
		return f
	}

	for len(data) > 0 {
		line, data, _ = bytes.Cut(data, newline)
		addr := next()
		loStr, hiStr, ok := strings.Cut(string(addr), "-")
		if !ok {
			continue
		}
		lo, err := strconv.ParseUint(loStr, 16, 64)
		if err != nil {
			continue
		}
		hi, err := strconv.ParseUint(hiStr, 16, 64)
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

		// Trim deleted file marker.
		deletedStr := " (deleted)"
		deletedLen := len(deletedStr)
		if len(file) >= deletedLen && file[len(file)-deletedLen:] == deletedStr {
			file = file[:len(file)-deletedLen]
		}

		if len(inode) == 1 && inode[0] == '0' && file == "" {
			// Huge-page text mappings list the initial fragment of
			// mapped but unpopulated memory as being inode 0.
			// Don't report that part.
			// But [vdso] and [vsyscall] are inode 0, so let non-empty file names through.
			continue
		}

		// TODO: pprof's remapMappingIDs makes one adjustment:
		// 1. If there is an /anon_hugepage mapping first and it is
		// consecutive to a next mapping, drop the /anon_hugepage.
		// There's no indication why this is needed.
		// Let's try not doing this and see what breaks.
		// If we do need it, it would go here, before we
		// enter the mappings into b.mem in the first place.

		buildID, _ := elfBuildID(file)
		addMapping(lo, hi, offset, file, buildID)
	}
}

func (b *profileBuilder) addMapping(lo, hi, offset uint64, file, buildID string) {
	b.addMappingEntry(lo, hi, offset, file, buildID, false)
}

func (b *profileBuilder) addMappingEntry(lo, hi, offset uint64, file, buildID string, fake bool) {
	b.mem = append(b.mem, memMap{
		start:   uintptr(lo),
		end:     uintptr(hi),
		offset:  offset,
		file:    file,
		buildID: buildID,
		fake:    fake,
	})
}
