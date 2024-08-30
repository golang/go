// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package report

// This file contains routines related to the generation of annotated
// source listings.

import (
	"bufio"
	"fmt"
	"html/template"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/google/pprof/internal/graph"
	"github.com/google/pprof/internal/measurement"
	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/profile"
)

// printSource prints an annotated source listing, include all
// functions with samples that match the regexp rpt.options.symbol.
// The sources are sorted by function name and then by filename to
// eliminate potential nondeterminism.
func printSource(w io.Writer, rpt *Report) error {
	o := rpt.options
	g := rpt.newGraph(nil)

	// Identify all the functions that match the regexp provided.
	// Group nodes for each matching function.
	var functions graph.Nodes
	functionNodes := make(map[string]graph.Nodes)
	for _, n := range g.Nodes {
		if !o.Symbol.MatchString(n.Info.Name) {
			continue
		}
		if functionNodes[n.Info.Name] == nil {
			functions = append(functions, n)
		}
		functionNodes[n.Info.Name] = append(functionNodes[n.Info.Name], n)
	}
	functions.Sort(graph.NameOrder)

	if len(functionNodes) == 0 {
		return fmt.Errorf("no matches found for regexp: %s", o.Symbol)
	}

	sourcePath := o.SourcePath
	if sourcePath == "" {
		wd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("could not stat current dir: %v", err)
		}
		sourcePath = wd
	}
	reader := newSourceReader(sourcePath, o.TrimPath)

	fmt.Fprintf(w, "Total: %s\n", rpt.formatValue(rpt.total))
	for _, fn := range functions {
		name := fn.Info.Name

		// Identify all the source files associated to this function.
		// Group nodes for each source file.
		var sourceFiles graph.Nodes
		fileNodes := make(map[string]graph.Nodes)
		for _, n := range functionNodes[name] {
			if n.Info.File == "" {
				continue
			}
			if fileNodes[n.Info.File] == nil {
				sourceFiles = append(sourceFiles, n)
			}
			fileNodes[n.Info.File] = append(fileNodes[n.Info.File], n)
		}

		if len(sourceFiles) == 0 {
			fmt.Fprintf(w, "No source information for %s\n", name)
			continue
		}

		sourceFiles.Sort(graph.FileOrder)

		// Print each file associated with this function.
		for _, fl := range sourceFiles {
			filename := fl.Info.File
			fns := fileNodes[filename]
			flatSum, cumSum := fns.Sum()

			fnodes, _, err := getSourceFromFile(filename, reader, fns, 0, 0)
			fmt.Fprintf(w, "ROUTINE ======================== %s in %s\n", name, filename)
			fmt.Fprintf(w, "%10s %10s (flat, cum) %s of Total\n",
				rpt.formatValue(flatSum), rpt.formatValue(cumSum),
				measurement.Percentage(cumSum, rpt.total))

			if err != nil {
				fmt.Fprintf(w, " Error: %v\n", err)
				continue
			}

			for _, fn := range fnodes {
				fmt.Fprintf(w, "%10s %10s %6d:%s\n", valueOrDot(fn.Flat, rpt), valueOrDot(fn.Cum, rpt), fn.Info.Lineno, fn.Info.Name)
			}
		}
	}
	return nil
}

// sourcePrinter holds state needed for generating source+asm HTML listing.
type sourcePrinter struct {
	reader     *sourceReader
	synth      *synthCode
	objectTool plugin.ObjTool
	objects    map[string]plugin.ObjFile  // Opened object files
	sym        *regexp.Regexp             // May be nil
	files      map[string]*sourceFile     // Set of files to print.
	insts      map[uint64]instructionInfo // Instructions of interest (keyed by address).

	// Set of function names that we are interested in (because they had
	// a sample and match sym).
	interest map[string]bool

	// Mapping from system function names to printable names.
	prettyNames map[string]string
}

// addrInfo holds information for an address we are interested in.
type addrInfo struct {
	loc *profile.Location // Always non-nil
	obj plugin.ObjFile    // May be nil
}

// instructionInfo holds collected information for an instruction.
type instructionInfo struct {
	objAddr   uint64 // Address in object file (with base subtracted out)
	length    int    // Instruction length in bytes
	disasm    string // Disassembly of instruction
	file      string // For top-level function in which instruction occurs
	line      int    // For top-level function in which instruction occurs
	flat, cum int64  // Samples to report (divisor already applied)
}

// sourceFile contains collected information for files we will print.
type sourceFile struct {
	fname    string
	cum      int64
	flat     int64
	lines    map[int][]sourceInst // Instructions to show per line
	funcName map[int]string       // Function name per line
}

// sourceInst holds information for an instruction to be displayed.
type sourceInst struct {
	addr  uint64
	stack []callID // Inlined call-stack
}

// sourceFunction contains information for a contiguous range of lines per function we
// will print.
type sourceFunction struct {
	name       string
	begin, end int // Line numbers (end is not included in the range)
	flat, cum  int64
}

// addressRange is a range of addresses plus the object file that contains it.
type addressRange struct {
	begin, end uint64
	obj        plugin.ObjFile
	mapping    *profile.Mapping
	score      int64 // Used to order ranges for processing
}

// WebListData holds the data needed to generate HTML source code listing.
type WebListData struct {
	Total string
	Files []WebListFile
}

// WebListFile holds the per-file information for HTML source code listing.
type WebListFile struct {
	Funcs []WebListFunc
}

// WebListFunc holds the per-function information for HTML source code listing.
type WebListFunc struct {
	Name       string
	File       string
	Flat       string
	Cumulative string
	Percent    string
	Lines      []WebListLine
}

// WebListLine holds the per-source-line information for HTML source code listing.
type WebListLine struct {
	SrcLine      string
	HTMLClass    string
	Line         int
	Flat         string
	Cumulative   string
	Instructions []WebListInstruction
}

// WebListInstruction holds the per-instruction information for HTML source code listing.
type WebListInstruction struct {
	NewBlock     bool // Insert marker that indicates separation from previous block
	Flat         string
	Cumulative   string
	Synthetic    bool
	Address      uint64
	Disasm       string
	FileLine     string
	InlinedCalls []WebListCall
}

// WebListCall holds the per-inlined-call information for HTML source code listing.
type WebListCall struct {
	SrcLine  string
	FileBase string
	Line     int
}

// MakeWebList returns an annotated source listing of rpt.
// rpt.prof should contain inlined call info.
func MakeWebList(rpt *Report, obj plugin.ObjTool, maxFiles int) (WebListData, error) {
	sourcePath := rpt.options.SourcePath
	if sourcePath == "" {
		wd, err := os.Getwd()
		if err != nil {
			return WebListData{}, fmt.Errorf("could not stat current dir: %v", err)
		}
		sourcePath = wd
	}
	sp := newSourcePrinter(rpt, obj, sourcePath)
	if len(sp.interest) == 0 {
		return WebListData{}, fmt.Errorf("no matches found for regexp: %s", rpt.options.Symbol)
	}
	defer sp.close()
	return sp.generate(maxFiles, rpt), nil
}

func newSourcePrinter(rpt *Report, obj plugin.ObjTool, sourcePath string) *sourcePrinter {
	sp := &sourcePrinter{
		reader:      newSourceReader(sourcePath, rpt.options.TrimPath),
		synth:       newSynthCode(rpt.prof.Mapping),
		objectTool:  obj,
		objects:     map[string]plugin.ObjFile{},
		sym:         rpt.options.Symbol,
		files:       map[string]*sourceFile{},
		insts:       map[uint64]instructionInfo{},
		prettyNames: map[string]string{},
		interest:    map[string]bool{},
	}

	// If the regexp source can be parsed as an address, also match
	// functions that land on that address.
	var address *uint64
	if sp.sym != nil {
		if hex, err := strconv.ParseUint(sp.sym.String(), 0, 64); err == nil {
			address = &hex
		}
	}

	addrs := map[uint64]addrInfo{}
	flat := map[uint64]int64{}
	cum := map[uint64]int64{}

	// Record an interest in the function corresponding to lines[index].
	markInterest := func(addr uint64, loc *profile.Location, index int) {
		fn := loc.Line[index]
		if fn.Function == nil {
			return
		}
		sp.interest[fn.Function.Name] = true
		sp.interest[fn.Function.SystemName] = true
		if _, ok := addrs[addr]; !ok {
			addrs[addr] = addrInfo{loc, sp.objectFile(loc.Mapping)}
		}
	}

	// See if sp.sym matches line.
	matches := func(line profile.Line) bool {
		if line.Function == nil {
			return false
		}
		return sp.sym.MatchString(line.Function.Name) ||
			sp.sym.MatchString(line.Function.SystemName) ||
			sp.sym.MatchString(line.Function.Filename)
	}

	// Extract sample counts and compute set of interesting functions.
	for _, sample := range rpt.prof.Sample {
		value := rpt.options.SampleValue(sample.Value)
		if rpt.options.SampleMeanDivisor != nil {
			div := rpt.options.SampleMeanDivisor(sample.Value)
			if div != 0 {
				value /= div
			}
		}

		// Find call-sites matching sym.
		for i := len(sample.Location) - 1; i >= 0; i-- {
			loc := sample.Location[i]
			for _, line := range loc.Line {
				if line.Function == nil {
					continue
				}
				sp.prettyNames[line.Function.SystemName] = line.Function.Name
			}

			addr := loc.Address
			if addr == 0 {
				// Some profiles are missing valid addresses.
				addr = sp.synth.address(loc)
			}

			cum[addr] += value
			if i == 0 {
				flat[addr] += value
			}

			if sp.sym == nil || (address != nil && addr == *address) {
				// Interested in top-level entry of stack.
				if len(loc.Line) > 0 {
					markInterest(addr, loc, len(loc.Line)-1)
				}
				continue
			}

			// Search in inlined stack for a match.
			matchFile := (loc.Mapping != nil && sp.sym.MatchString(loc.Mapping.File))
			for j, line := range loc.Line {
				if (j == 0 && matchFile) || matches(line) {
					markInterest(addr, loc, j)
				}
			}
		}
	}

	sp.expandAddresses(rpt, addrs, flat)
	sp.initSamples(flat, cum)
	return sp
}

func (sp *sourcePrinter) close() {
	for _, objFile := range sp.objects {
		if objFile != nil {
			objFile.Close()
		}
	}
}

func (sp *sourcePrinter) expandAddresses(rpt *Report, addrs map[uint64]addrInfo, flat map[uint64]int64) {
	// We found interesting addresses (ones with non-zero samples) above.
	// Get covering address ranges and disassemble the ranges.
	ranges, unprocessed := sp.splitIntoRanges(rpt.prof, addrs, flat)
	sp.handleUnprocessed(addrs, unprocessed)

	// Trim ranges if there are too many.
	const maxRanges = 25
	sort.Slice(ranges, func(i, j int) bool {
		return ranges[i].score > ranges[j].score
	})
	if len(ranges) > maxRanges {
		ranges = ranges[:maxRanges]
	}

	for _, r := range ranges {
		objBegin, err := r.obj.ObjAddr(r.begin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to compute objdump address for range start %x: %v\n", r.begin, err)
			continue
		}
		objEnd, err := r.obj.ObjAddr(r.end)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to compute objdump address for range end %x: %v\n", r.end, err)
			continue
		}
		base := r.begin - objBegin
		insts, err := sp.objectTool.Disasm(r.mapping.File, objBegin, objEnd, rpt.options.IntelSyntax)
		if err != nil {
			// TODO(sanjay): Report that the covered addresses are missing.
			continue
		}

		var lastFrames []plugin.Frame
		var lastAddr, maxAddr uint64
		for i, inst := range insts {
			addr := inst.Addr + base

			// Guard against duplicate output from Disasm.
			if addr <= maxAddr {
				continue
			}
			maxAddr = addr

			length := 1
			if i+1 < len(insts) && insts[i+1].Addr > inst.Addr {
				// Extend to next instruction.
				length = int(insts[i+1].Addr - inst.Addr)
			}

			// Get inlined-call-stack for address.
			frames, err := r.obj.SourceLine(addr)
			if err != nil {
				// Construct a frame from disassembler output.
				frames = []plugin.Frame{{Func: inst.Function, File: inst.File, Line: inst.Line}}
			}

			x := instructionInfo{objAddr: inst.Addr, length: length, disasm: inst.Text}
			if len(frames) > 0 {
				// We could consider using the outer-most caller's source
				// location so we give the some hint as to where the
				// inlining happened that led to this instruction. So for
				// example, suppose we have the following (inlined) call
				// chains for this instruction:
				//   F1->G->H
				//   F2->G->H
				// We could tag the instructions from the first call with
				// F1 and instructions from the second call with F2. But
				// that leads to a somewhat confusing display. So for now,
				// we stick with just the inner-most location (i.e., H).
				// In the future we will consider changing the display to
				// make caller info more visible.
				index := 0 // Inner-most frame
				x.file = frames[index].File
				x.line = frames[index].Line
			}
			sp.insts[addr] = x

			// We sometimes get instructions with a zero reported line number.
			// Make such instructions have the same line info as the preceding
			// instruction, if an earlier instruction is found close enough.
			const neighborhood = 32
			if len(frames) > 0 && frames[0].Line != 0 {
				lastFrames = frames
				lastAddr = addr
			} else if (addr-lastAddr <= neighborhood) && lastFrames != nil {
				frames = lastFrames
			}

			sp.addStack(addr, frames)
		}
	}
}

func (sp *sourcePrinter) addStack(addr uint64, frames []plugin.Frame) {
	// See if the stack contains a function we are interested in.
	for i, f := range frames {
		if !sp.interest[f.Func] {
			continue
		}

		// Record sub-stack under frame's file/line.
		fname := canonicalizeFileName(f.File)
		file := sp.files[fname]
		if file == nil {
			file = &sourceFile{
				fname:    fname,
				lines:    map[int][]sourceInst{},
				funcName: map[int]string{},
			}
			sp.files[fname] = file
		}
		callees := frames[:i]
		stack := make([]callID, 0, len(callees))
		for j := len(callees) - 1; j >= 0; j-- { // Reverse so caller is first
			stack = append(stack, callID{
				file: callees[j].File,
				line: callees[j].Line,
			})
		}
		file.lines[f.Line] = append(file.lines[f.Line], sourceInst{addr, stack})

		// Remember the first function name encountered per source line
		// and assume that that line belongs to that function.
		if _, ok := file.funcName[f.Line]; !ok {
			file.funcName[f.Line] = f.Func
		}
	}
}

// synthAsm is the special disassembler value used for instructions without an object file.
const synthAsm = ""

// handleUnprocessed handles addresses that were skipped by splitIntoRanges because they
// did not belong to a known object file.
func (sp *sourcePrinter) handleUnprocessed(addrs map[uint64]addrInfo, unprocessed []uint64) {
	// makeFrames synthesizes a []plugin.Frame list for the specified address.
	// The result will typically have length 1, but may be longer if address corresponds
	// to inlined calls.
	makeFrames := func(addr uint64) []plugin.Frame {
		loc := addrs[addr].loc
		stack := make([]plugin.Frame, 0, len(loc.Line))
		for _, line := range loc.Line {
			fn := line.Function
			if fn == nil {
				continue
			}
			stack = append(stack, plugin.Frame{
				Func: fn.Name,
				File: fn.Filename,
				Line: int(line.Line),
			})
		}
		return stack
	}

	for _, addr := range unprocessed {
		frames := makeFrames(addr)
		x := instructionInfo{
			objAddr: addr,
			length:  1,
			disasm:  synthAsm,
		}
		if len(frames) > 0 {
			x.file = frames[0].File
			x.line = frames[0].Line
		}
		sp.insts[addr] = x

		sp.addStack(addr, frames)
	}
}

// splitIntoRanges converts the set of addresses we are interested in into a set of address
// ranges to disassemble. It also returns the set of addresses found that did not have an
// associated object file and were therefore not added to an address range.
func (sp *sourcePrinter) splitIntoRanges(prof *profile.Profile, addrMap map[uint64]addrInfo, flat map[uint64]int64) ([]addressRange, []uint64) {
	// Partition addresses into two sets: ones with a known object file, and ones without.
	var addrs, unprocessed []uint64
	for addr, info := range addrMap {
		if info.obj != nil {
			addrs = append(addrs, addr)
		} else {
			unprocessed = append(unprocessed, addr)
		}
	}
	sort.Slice(addrs, func(i, j int) bool { return addrs[i] < addrs[j] })

	const expand = 500 // How much to expand range to pick up nearby addresses.
	var result []addressRange
	for i, n := 0, len(addrs); i < n; {
		begin, end := addrs[i], addrs[i]
		sum := flat[begin]
		i++

		info := addrMap[begin]
		m := info.loc.Mapping
		obj := info.obj // Non-nil because of the partitioning done above.

		// Find following addresses that are close enough to addrs[i].
		for i < n && addrs[i] <= end+2*expand && addrs[i] < m.Limit {
			// When we expand ranges by "expand" on either side, the ranges
			// for addrs[i] and addrs[i-1] will merge.
			end = addrs[i]
			sum += flat[end]
			i++
		}
		if m.Start-begin >= expand {
			begin -= expand
		} else {
			begin = m.Start
		}
		if m.Limit-end >= expand {
			end += expand
		} else {
			end = m.Limit
		}

		result = append(result, addressRange{begin, end, obj, m, sum})
	}
	return result, unprocessed
}

func (sp *sourcePrinter) initSamples(flat, cum map[uint64]int64) {
	for addr, inst := range sp.insts {
		// Move all samples that were assigned to the middle of an instruction to the
		// beginning of that instruction. This takes care of samples that were recorded
		// against pc+1.
		instEnd := addr + uint64(inst.length)
		for p := addr; p < instEnd; p++ {
			inst.flat += flat[p]
			inst.cum += cum[p]
		}
		sp.insts[addr] = inst
	}
}

func (sp *sourcePrinter) generate(maxFiles int, rpt *Report) WebListData {
	// Finalize per-file counts.
	for _, file := range sp.files {
		seen := map[uint64]bool{}
		for _, line := range file.lines {
			for _, x := range line {
				if seen[x.addr] {
					// Same address can be displayed multiple times in a file
					// (e.g., if we show multiple inlined functions).
					// Avoid double-counting samples in this case.
					continue
				}
				seen[x.addr] = true
				inst := sp.insts[x.addr]
				file.cum += inst.cum
				file.flat += inst.flat
			}
		}
	}

	// Get sorted list of files to print.
	var files []*sourceFile
	for _, f := range sp.files {
		files = append(files, f)
	}
	order := func(i, j int) bool { return files[i].flat > files[j].flat }
	if maxFiles < 0 {
		// Order by name for compatibility with old code.
		order = func(i, j int) bool { return files[i].fname < files[j].fname }
		maxFiles = len(files)
	}
	sort.Slice(files, order)
	result := WebListData{
		Total: rpt.formatValue(rpt.total),
	}
	for i, f := range files {
		if i < maxFiles {
			result.Files = append(result.Files, sp.generateFile(f, rpt))
		}
	}
	return result
}

func (sp *sourcePrinter) generateFile(f *sourceFile, rpt *Report) WebListFile {
	var result WebListFile
	for _, fn := range sp.functions(f) {
		if fn.cum == 0 {
			continue
		}

		listfn := WebListFunc{
			Name:       fn.name,
			File:       f.fname,
			Flat:       rpt.formatValue(fn.flat),
			Cumulative: rpt.formatValue(fn.cum),
			Percent:    measurement.Percentage(fn.cum, rpt.total),
		}
		var asm []assemblyInstruction
		for l := fn.begin; l < fn.end; l++ {
			lineContents, ok := sp.reader.line(f.fname, l)
			if !ok {
				if len(f.lines[l]) == 0 {
					// Outside of range of valid lines and nothing to print.
					continue
				}
				if l == 0 {
					// Line number 0 shows up if line number is not known.
					lineContents = "<instructions with unknown line numbers>"
				} else {
					// Past end of file, but have data to print.
					lineContents = "???"
				}
			}

			// Make list of assembly instructions.
			asm = asm[:0]
			var flatSum, cumSum int64
			var lastAddr uint64
			for _, inst := range f.lines[l] {
				addr := inst.addr
				x := sp.insts[addr]
				flatSum += x.flat
				cumSum += x.cum
				startsBlock := (addr != lastAddr+uint64(sp.insts[lastAddr].length))
				lastAddr = addr

				// divisors already applied, so leave flatDiv,cumDiv as 0
				asm = append(asm, assemblyInstruction{
					address:     x.objAddr,
					instruction: x.disasm,
					function:    fn.name,
					file:        x.file,
					line:        x.line,
					flat:        x.flat,
					cum:         x.cum,
					startsBlock: startsBlock,
					inlineCalls: inst.stack,
				})
			}

			listfn.Lines = append(listfn.Lines, makeWebListLine(l, flatSum, cumSum, lineContents, asm, sp.reader, rpt))
		}

		result.Funcs = append(result.Funcs, listfn)
	}
	return result
}

// functions splits apart the lines to show in a file into a list of per-function ranges.
func (sp *sourcePrinter) functions(f *sourceFile) []sourceFunction {
	var funcs []sourceFunction

	// Get interesting lines in sorted order.
	lines := make([]int, 0, len(f.lines))
	for l := range f.lines {
		lines = append(lines, l)
	}
	sort.Ints(lines)

	// Merge adjacent lines that are in same function and not too far apart.
	const mergeLimit = 20
	for _, l := range lines {
		name := f.funcName[l]
		if pretty, ok := sp.prettyNames[name]; ok {
			// Use demangled name if available.
			name = pretty
		}

		fn := sourceFunction{name: name, begin: l, end: l + 1}
		for _, x := range f.lines[l] {
			inst := sp.insts[x.addr]
			fn.flat += inst.flat
			fn.cum += inst.cum
		}

		// See if we should merge into preceding function.
		if len(funcs) > 0 {
			last := funcs[len(funcs)-1]
			if l-last.end < mergeLimit && last.name == name {
				last.end = l + 1
				last.flat += fn.flat
				last.cum += fn.cum
				funcs[len(funcs)-1] = last
				continue
			}
		}

		// Add new function.
		funcs = append(funcs, fn)
	}

	// Expand function boundaries to show neighborhood.
	const expand = 5
	for i, f := range funcs {
		if i == 0 {
			// Extend backwards, stopping at line number 1, but do not disturb 0
			// since that is a special line number that can show up when addr2line
			// cannot determine the real line number.
			if f.begin > expand {
				f.begin -= expand
			} else if f.begin > 1 {
				f.begin = 1
			}
		} else {
			// Find gap from predecessor and divide between predecessor and f.
			halfGap := (f.begin - funcs[i-1].end) / 2
			if halfGap > expand {
				halfGap = expand
			}
			funcs[i-1].end += halfGap
			f.begin -= halfGap
		}
		funcs[i] = f
	}

	// Also extend the ending point of the last function.
	if len(funcs) > 0 {
		funcs[len(funcs)-1].end += expand
	}

	return funcs
}

// objectFile return the object for the specified mapping, opening it if necessary.
// It returns nil on error.
func (sp *sourcePrinter) objectFile(m *profile.Mapping) plugin.ObjFile {
	if m == nil {
		return nil
	}
	if object, ok := sp.objects[m.File]; ok {
		return object // May be nil if we detected an error earlier.
	}
	object, err := sp.objectTool.Open(m.File, m.Start, m.Limit, m.Offset, m.KernelRelocationSymbol)
	if err != nil {
		object = nil
	}
	sp.objects[m.File] = object // Cache even on error.
	return object
}

// makeWebListLine returns the contents of a single line in a web listing. This includes
// the source line and the corresponding assembly.
func makeWebListLine(lineNo int, flat, cum int64, lineContents string,
	assembly []assemblyInstruction, reader *sourceReader, rpt *Report) WebListLine {
	line := WebListLine{
		SrcLine:    lineContents,
		Line:       lineNo,
		Flat:       valueOrDot(flat, rpt),
		Cumulative: valueOrDot(cum, rpt),
	}

	if len(assembly) == 0 {
		line.HTMLClass = "nop"
		return line
	}

	nestedInfo := false
	line.HTMLClass = "deadsrc"
	for _, an := range assembly {
		if len(an.inlineCalls) > 0 || an.instruction != synthAsm {
			nestedInfo = true
			line.HTMLClass = "livesrc"
		}
	}

	if nestedInfo {
		srcIndent := indentation(lineContents)
		line.Instructions = makeWebListInstructions(srcIndent, assembly, reader, rpt)
	}
	return line
}

func makeWebListInstructions(srcIndent int, assembly []assemblyInstruction, reader *sourceReader, rpt *Report) []WebListInstruction {
	var result []WebListInstruction
	var curCalls []callID
	for i, an := range assembly {
		var fileline string
		if an.file != "" {
			fileline = fmt.Sprintf("%s:%d", template.HTMLEscapeString(filepath.Base(an.file)), an.line)
		}
		text := strings.Repeat(" ", srcIndent+4+4*len(an.inlineCalls)) + an.instruction
		inst := WebListInstruction{
			NewBlock:   (an.startsBlock && i != 0),
			Flat:       valueOrDot(an.flat, rpt),
			Cumulative: valueOrDot(an.cum, rpt),
			Synthetic:  (an.instruction == synthAsm),
			Address:    an.address,
			Disasm:     rightPad(text, 80),
			FileLine:   fileline,
		}

		// Add inlined call context.
		for j, c := range an.inlineCalls {
			if j < len(curCalls) && curCalls[j] == c {
				// Skip if same as previous instruction.
				continue
			}
			curCalls = nil
			fline, ok := reader.line(c.file, c.line)
			if !ok {
				fline = ""
			}
			srcCode := strings.Repeat(" ", srcIndent+4+4*j) + strings.TrimSpace(fline)
			inst.InlinedCalls = append(inst.InlinedCalls, WebListCall{
				SrcLine:  rightPad(srcCode, 80),
				FileBase: filepath.Base(c.file),
				Line:     c.line,
			})
		}
		curCalls = an.inlineCalls

		result = append(result, inst)
	}
	return result
}

// getSourceFromFile collects the sources of a function from a source
// file and annotates it with the samples in fns. Returns the sources
// as nodes, using the info.name field to hold the source code.
func getSourceFromFile(file string, reader *sourceReader, fns graph.Nodes, start, end int) (graph.Nodes, string, error) {
	lineNodes := make(map[int]graph.Nodes)

	// Collect source coordinates from profile.
	const margin = 5 // Lines before first/after last sample.
	if start == 0 {
		if fns[0].Info.StartLine != 0 {
			start = fns[0].Info.StartLine
		} else {
			start = fns[0].Info.Lineno - margin
		}
	} else {
		start -= margin
	}
	if end == 0 {
		end = fns[0].Info.Lineno
	}
	end += margin
	for _, n := range fns {
		lineno := n.Info.Lineno
		nodeStart := n.Info.StartLine
		if nodeStart == 0 {
			nodeStart = lineno - margin
		}
		nodeEnd := lineno + margin
		if nodeStart < start {
			start = nodeStart
		} else if nodeEnd > end {
			end = nodeEnd
		}
		lineNodes[lineno] = append(lineNodes[lineno], n)
	}
	if start < 1 {
		start = 1
	}

	var src graph.Nodes
	for lineno := start; lineno <= end; lineno++ {
		line, ok := reader.line(file, lineno)
		if !ok {
			break
		}
		flat, cum := lineNodes[lineno].Sum()
		src = append(src, &graph.Node{
			Info: graph.NodeInfo{
				Name:   strings.TrimRight(line, "\n"),
				Lineno: lineno,
			},
			Flat: flat,
			Cum:  cum,
		})
	}
	if err := reader.fileError(file); err != nil {
		return nil, file, err
	}
	return src, file, nil
}

// sourceReader provides access to source code with caching of file contents.
type sourceReader struct {
	// searchPath is a filepath.ListSeparator-separated list of directories where
	// source files should be searched.
	searchPath string

	// trimPath is a filepath.ListSeparator-separated list of paths to trim.
	trimPath string

	// files maps from path name to a list of lines.
	// files[*][0] is unused since line numbering starts at 1.
	files map[string][]string

	// errors collects errors encountered per file. These errors are
	// consulted before returning out of these module.
	errors map[string]error
}

func newSourceReader(searchPath, trimPath string) *sourceReader {
	return &sourceReader{
		searchPath,
		trimPath,
		make(map[string][]string),
		make(map[string]error),
	}
}

func (reader *sourceReader) fileError(path string) error {
	return reader.errors[path]
}

// line returns the line numbered "lineno" in path, or _,false if lineno is out of range.
func (reader *sourceReader) line(path string, lineno int) (string, bool) {
	lines, ok := reader.files[path]
	if !ok {
		// Read and cache file contents.
		lines = []string{""} // Skip 0th line
		f, err := openSourceFile(path, reader.searchPath, reader.trimPath)
		if err != nil {
			reader.errors[path] = err
		} else {
			s := bufio.NewScanner(f)
			for s.Scan() {
				lines = append(lines, s.Text())
			}
			f.Close()
			if s.Err() != nil {
				reader.errors[path] = err
			}
		}
		reader.files[path] = lines
	}
	if lineno <= 0 || lineno >= len(lines) {
		return "", false
	}
	return lines[lineno], true
}

// openSourceFile opens a source file from a name encoded in a profile. File
// names in a profile after can be relative paths, so search them in each of
// the paths in searchPath and their parents. In case the profile contains
// absolute paths, additional paths may be configured to trim from the source
// paths in the profile. This effectively turns the path into a relative path
// searching it using searchPath as usual).
func openSourceFile(path, searchPath, trim string) (*os.File, error) {
	path = trimPath(path, trim, searchPath)
	// If file is still absolute, require file to exist.
	if filepath.IsAbs(path) {
		f, err := os.Open(path)
		return f, err
	}
	// Scan each component of the path.
	for _, dir := range filepath.SplitList(searchPath) {
		// Search up for every parent of each possible path.
		for {
			filename := filepath.Join(dir, path)
			if f, err := os.Open(filename); err == nil {
				return f, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	return nil, fmt.Errorf("could not find file %s on path %s", path, searchPath)
}

// trimPath cleans up a path by removing prefixes that are commonly
// found on profiles plus configured prefixes.
// TODO(aalexand): Consider optimizing out the redundant work done in this
// function if it proves to matter.
func trimPath(path, trimPath, searchPath string) string {
	// Keep path variable intact as it's used below to form the return value.
	sPath, searchPath := filepath.ToSlash(path), filepath.ToSlash(searchPath)
	if trimPath == "" {
		// If the trim path is not configured, try to guess it heuristically:
		// search for basename of each search path in the original path and, if
		// found, strip everything up to and including the basename. So, for
		// example, given original path "/some/remote/path/my-project/foo/bar.c"
		// and search path "/my/local/path/my-project" the heuristic will return
		// "/my/local/path/my-project/foo/bar.c".
		for _, dir := range filepath.SplitList(searchPath) {
			want := "/" + filepath.Base(dir) + "/"
			if found := strings.Index(sPath, want); found != -1 {
				return path[found+len(want):]
			}
		}
	}
	// Trim configured trim prefixes.
	trimPaths := append(filepath.SplitList(filepath.ToSlash(trimPath)), "/proc/self/cwd/./", "/proc/self/cwd/")
	for _, trimPath := range trimPaths {
		if !strings.HasSuffix(trimPath, "/") {
			trimPath += "/"
		}
		if strings.HasPrefix(sPath, trimPath) {
			return path[len(trimPath):]
		}
	}
	return path
}

func indentation(line string) int {
	column := 0
	for _, c := range line {
		if c == ' ' {
			column++
		} else if c == '\t' {
			column++
			for column%8 != 0 {
				column++
			}
		} else {
			break
		}
	}
	return column
}

// rightPad pads the input with spaces on the right-hand-side to make it have
// at least width n. It treats tabs as enough spaces that lead to the next
// 8-aligned tab-stop.
func rightPad(s string, n int) string {
	var str strings.Builder

	// Convert tabs to spaces as we go so padding works regardless of what prefix
	// is placed before the result.
	column := 0
	for _, c := range s {
		column++
		if c == '\t' {
			str.WriteRune(' ')
			for column%8 != 0 {
				column++
				str.WriteRune(' ')
			}
		} else {
			str.WriteRune(c)
		}
	}
	for column < n {
		column++
		str.WriteRune(' ')
	}
	return str.String()
}

func canonicalizeFileName(fname string) string {
	fname = strings.TrimPrefix(fname, "/proc/self/cwd/")
	fname = strings.TrimPrefix(fname, "./")
	return filepath.Clean(fname)
}
