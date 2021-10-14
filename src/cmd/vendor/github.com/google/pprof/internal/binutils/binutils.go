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

// Package binutils provides access to the GNU binutils.
package binutils

import (
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/google/pprof/internal/elfexec"
	"github.com/google/pprof/internal/plugin"
)

// A Binutils implements plugin.ObjTool by invoking the GNU binutils.
type Binutils struct {
	mu  sync.Mutex
	rep *binrep
}

var (
	objdumpLLVMVerRE = regexp.MustCompile(`LLVM version (?:(\d*)\.(\d*)\.(\d*)|.*(trunk).*)`)

	// Defined for testing
	elfOpen = elf.Open
)

// binrep is an immutable representation for Binutils.  It is atomically
// replaced on every mutation to provide thread-safe access.
type binrep struct {
	// Commands to invoke.
	llvmSymbolizer      string
	llvmSymbolizerFound bool
	addr2line           string
	addr2lineFound      bool
	nm                  string
	nmFound             bool
	objdump             string
	objdumpFound        bool
	isLLVMObjdump       bool

	// if fast, perform symbolization using nm (symbol names only),
	// instead of file-line detail from the slower addr2line.
	fast bool
}

// get returns the current representation for bu, initializing it if necessary.
func (bu *Binutils) get() *binrep {
	bu.mu.Lock()
	r := bu.rep
	if r == nil {
		r = &binrep{}
		initTools(r, "")
		bu.rep = r
	}
	bu.mu.Unlock()
	return r
}

// update modifies the rep for bu via the supplied function.
func (bu *Binutils) update(fn func(r *binrep)) {
	r := &binrep{}
	bu.mu.Lock()
	defer bu.mu.Unlock()
	if bu.rep == nil {
		initTools(r, "")
	} else {
		*r = *bu.rep
	}
	fn(r)
	bu.rep = r
}

// String returns string representation of the binutils state for debug logging.
func (bu *Binutils) String() string {
	r := bu.get()
	var llvmSymbolizer, addr2line, nm, objdump string
	if r.llvmSymbolizerFound {
		llvmSymbolizer = r.llvmSymbolizer
	}
	if r.addr2lineFound {
		addr2line = r.addr2line
	}
	if r.nmFound {
		nm = r.nm
	}
	if r.objdumpFound {
		objdump = r.objdump
	}
	return fmt.Sprintf("llvm-symbolizer=%q addr2line=%q nm=%q objdump=%q fast=%t",
		llvmSymbolizer, addr2line, nm, objdump, r.fast)
}

// SetFastSymbolization sets a toggle that makes binutils use fast
// symbolization (using nm), which is much faster than addr2line but
// provides only symbol name information (no file/line).
func (bu *Binutils) SetFastSymbolization(fast bool) {
	bu.update(func(r *binrep) { r.fast = fast })
}

// SetTools processes the contents of the tools option. It
// expects a set of entries separated by commas; each entry is a pair
// of the form t:path, where cmd will be used to look only for the
// tool named t. If t is not specified, the path is searched for all
// tools.
func (bu *Binutils) SetTools(config string) {
	bu.update(func(r *binrep) { initTools(r, config) })
}

func initTools(b *binrep, config string) {
	// paths collect paths per tool; Key "" contains the default.
	paths := make(map[string][]string)
	for _, t := range strings.Split(config, ",") {
		name, path := "", t
		if ct := strings.SplitN(t, ":", 2); len(ct) == 2 {
			name, path = ct[0], ct[1]
		}
		paths[name] = append(paths[name], path)
	}

	defaultPath := paths[""]
	b.llvmSymbolizer, b.llvmSymbolizerFound = chooseExe([]string{"llvm-symbolizer"}, []string{}, append(paths["llvm-symbolizer"], defaultPath...))
	b.addr2line, b.addr2lineFound = chooseExe([]string{"addr2line"}, []string{"gaddr2line"}, append(paths["addr2line"], defaultPath...))
	// The "-n" option is supported by LLVM since 2011. The output of llvm-nm
	// and GNU nm with "-n" option is interchangeable for our purposes, so we do
	// not need to differrentiate them.
	b.nm, b.nmFound = chooseExe([]string{"llvm-nm", "nm"}, []string{"gnm"}, append(paths["nm"], defaultPath...))
	b.objdump, b.objdumpFound, b.isLLVMObjdump = findObjdump(append(paths["objdump"], defaultPath...))
}

// findObjdump finds and returns path to preferred objdump binary.
// Order of preference is: llvm-objdump, objdump.
// On MacOS only, also looks for gobjdump with least preference.
// Accepts a list of paths and returns:
// a string with path to the preferred objdump binary if found,
// or an empty string if not found;
// a boolean if any acceptable objdump was found;
// a boolean indicating if it is an LLVM objdump.
func findObjdump(paths []string) (string, bool, bool) {
	objdumpNames := []string{"llvm-objdump", "objdump"}
	if runtime.GOOS == "darwin" {
		objdumpNames = append(objdumpNames, "gobjdump")
	}

	for _, objdumpName := range objdumpNames {
		if objdump, objdumpFound := findExe(objdumpName, paths); objdumpFound {
			cmdOut, err := exec.Command(objdump, "--version").Output()
			if err != nil {
				continue
			}
			if isLLVMObjdump(string(cmdOut)) {
				return objdump, true, true
			}
			if isBuObjdump(string(cmdOut)) {
				return objdump, true, false
			}
		}
	}
	return "", false, false
}

// chooseExe finds and returns path to preferred binary. names is a list of
// names to search on both Linux and OSX. osxNames is a list of names specific
// to OSX. names always has a higher priority than osxNames. The order of
// the name within each list decides its priority (e.g. the first name has a
// higher priority than the second name in the list).
//
// It returns a string with path to the binary and a boolean indicating if any
// acceptable binary was found.
func chooseExe(names, osxNames []string, paths []string) (string, bool) {
	if runtime.GOOS == "darwin" {
		names = append(names, osxNames...)
	}
	for _, name := range names {
		if binary, found := findExe(name, paths); found {
			return binary, true
		}
	}
	return "", false
}

// isLLVMObjdump accepts a string with path to an objdump binary,
// and returns a boolean indicating if the given binary is an LLVM
// objdump binary of an acceptable version.
func isLLVMObjdump(output string) bool {
	fields := objdumpLLVMVerRE.FindStringSubmatch(output)
	if len(fields) != 5 {
		return false
	}
	if fields[4] == "trunk" {
		return true
	}
	verMajor, err := strconv.Atoi(fields[1])
	if err != nil {
		return false
	}
	verPatch, err := strconv.Atoi(fields[3])
	if err != nil {
		return false
	}
	if runtime.GOOS == "linux" && verMajor >= 8 {
		// Ensure LLVM objdump is at least version 8.0 on Linux.
		// Some flags, like --demangle, and double dashes for options are
		// not supported by previous versions.
		return true
	}
	if runtime.GOOS == "darwin" {
		// Ensure LLVM objdump is at least version 10.0.1 on MacOS.
		return verMajor > 10 || (verMajor == 10 && verPatch >= 1)
	}
	return false
}

// isBuObjdump accepts a string with path to an objdump binary,
// and returns a boolean indicating if the given binary is a GNU
// binutils objdump binary. No version check is performed.
func isBuObjdump(output string) bool {
	return strings.Contains(output, "GNU objdump")
}

// findExe looks for an executable command on a set of paths.
// If it cannot find it, returns cmd.
func findExe(cmd string, paths []string) (string, bool) {
	for _, p := range paths {
		cp := filepath.Join(p, cmd)
		if c, err := exec.LookPath(cp); err == nil {
			return c, true
		}
	}
	return cmd, false
}

// Disasm returns the assembly instructions for the specified address range
// of a binary.
func (bu *Binutils) Disasm(file string, start, end uint64, intelSyntax bool) ([]plugin.Inst, error) {
	b := bu.get()
	if !b.objdumpFound {
		return nil, errors.New("cannot disasm: no objdump tool available")
	}
	args := []string{"--disassemble", "--demangle", "--no-show-raw-insn",
		"--line-numbers", fmt.Sprintf("--start-address=%#x", start),
		fmt.Sprintf("--stop-address=%#x", end)}

	if intelSyntax {
		if b.isLLVMObjdump {
			args = append(args, "--x86-asm-syntax=intel")
		} else {
			args = append(args, "-M", "intel")
		}
	}

	args = append(args, file)
	cmd := exec.Command(b.objdump, args...)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%v: %v", cmd.Args, err)
	}

	return disassemble(out)
}

// Open satisfies the plugin.ObjTool interface.
func (bu *Binutils) Open(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	b := bu.get()

	// Make sure file is a supported executable.
	// This uses magic numbers, mainly to provide better error messages but
	// it should also help speed.

	if _, err := os.Stat(name); err != nil {
		// For testing, do not require file name to exist.
		if strings.Contains(b.addr2line, "testdata/") {
			return &fileAddr2Line{file: file{b: b, name: name}}, nil
		}
		return nil, err
	}

	// Read the first 4 bytes of the file.

	f, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("error opening %s: %v", name, err)
	}
	defer f.Close()

	var header [4]byte
	if _, err = io.ReadFull(f, header[:]); err != nil {
		return nil, fmt.Errorf("error reading magic number from %s: %v", name, err)
	}

	elfMagic := string(header[:])

	// Match against supported file types.
	if elfMagic == elf.ELFMAG {
		f, err := b.openELF(name, start, limit, offset)
		if err != nil {
			return nil, fmt.Errorf("error reading ELF file %s: %v", name, err)
		}
		return f, nil
	}

	// Mach-O magic numbers can be big or little endian.
	machoMagicLittle := binary.LittleEndian.Uint32(header[:])
	machoMagicBig := binary.BigEndian.Uint32(header[:])

	if machoMagicLittle == macho.Magic32 || machoMagicLittle == macho.Magic64 ||
		machoMagicBig == macho.Magic32 || machoMagicBig == macho.Magic64 {
		f, err := b.openMachO(name, start, limit, offset)
		if err != nil {
			return nil, fmt.Errorf("error reading Mach-O file %s: %v", name, err)
		}
		return f, nil
	}
	if machoMagicLittle == macho.MagicFat || machoMagicBig == macho.MagicFat {
		f, err := b.openFatMachO(name, start, limit, offset)
		if err != nil {
			return nil, fmt.Errorf("error reading fat Mach-O file %s: %v", name, err)
		}
		return f, nil
	}

	peMagic := string(header[:2])
	if peMagic == "MZ" {
		f, err := b.openPE(name, start, limit, offset)
		if err != nil {
			return nil, fmt.Errorf("error reading PE file %s: %v", name, err)
		}
		return f, nil
	}

	return nil, fmt.Errorf("unrecognized binary format: %s", name)
}

func (b *binrep) openMachOCommon(name string, of *macho.File, start, limit, offset uint64) (plugin.ObjFile, error) {

	// Subtract the load address of the __TEXT section. Usually 0 for shared
	// libraries or 0x100000000 for executables. You can check this value by
	// running `objdump -private-headers <file>`.

	textSegment := of.Segment("__TEXT")
	if textSegment == nil {
		return nil, fmt.Errorf("could not identify base for %s: no __TEXT segment", name)
	}
	if textSegment.Addr > start {
		return nil, fmt.Errorf("could not identify base for %s: __TEXT segment address (0x%x) > mapping start address (0x%x)",
			name, textSegment.Addr, start)
	}

	base := start - textSegment.Addr

	if b.fast || (!b.addr2lineFound && !b.llvmSymbolizerFound) {
		return &fileNM{file: file{b: b, name: name, base: base}}, nil
	}
	return &fileAddr2Line{file: file{b: b, name: name, base: base}}, nil
}

func (b *binrep) openFatMachO(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	of, err := macho.OpenFat(name)
	if err != nil {
		return nil, fmt.Errorf("error parsing %s: %v", name, err)
	}
	defer of.Close()

	if len(of.Arches) == 0 {
		return nil, fmt.Errorf("empty fat Mach-O file: %s", name)
	}

	var arch macho.Cpu
	// Use the host architecture.
	// TODO: This is not ideal because the host architecture may not be the one
	// that was profiled. E.g. an amd64 host can profile a 386 program.
	switch runtime.GOARCH {
	case "386":
		arch = macho.Cpu386
	case "amd64", "amd64p32":
		arch = macho.CpuAmd64
	case "arm", "armbe", "arm64", "arm64be":
		arch = macho.CpuArm
	case "ppc":
		arch = macho.CpuPpc
	case "ppc64", "ppc64le":
		arch = macho.CpuPpc64
	default:
		return nil, fmt.Errorf("unsupported host architecture for %s: %s", name, runtime.GOARCH)
	}
	for i := range of.Arches {
		if of.Arches[i].Cpu == arch {
			return b.openMachOCommon(name, of.Arches[i].File, start, limit, offset)
		}
	}
	return nil, fmt.Errorf("architecture not found in %s: %s", name, runtime.GOARCH)
}

func (b *binrep) openMachO(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	of, err := macho.Open(name)
	if err != nil {
		return nil, fmt.Errorf("error parsing %s: %v", name, err)
	}
	defer of.Close()

	return b.openMachOCommon(name, of, start, limit, offset)
}

func (b *binrep) openELF(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	ef, err := elfOpen(name)
	if err != nil {
		return nil, fmt.Errorf("error parsing %s: %v", name, err)
	}
	defer ef.Close()

	buildID := ""
	if f, err := os.Open(name); err == nil {
		if id, err := elfexec.GetBuildID(f); err == nil {
			buildID = fmt.Sprintf("%x", id)
		}
	}

	var (
		stextOffset *uint64
		pageAligned = func(addr uint64) bool { return addr%4096 == 0 }
	)
	if strings.Contains(name, "vmlinux") || !pageAligned(start) || !pageAligned(limit) || !pageAligned(offset) {
		// Reading all Symbols is expensive, and we only rarely need it so
		// we don't want to do it every time. But if _stext happens to be
		// page-aligned but isn't the same as Vaddr, we would symbolize
		// wrong. So if the name the addresses aren't page aligned, or if
		// the name is "vmlinux" we read _stext. We can be wrong if: (1)
		// someone passes a kernel path that doesn't contain "vmlinux" AND
		// (2) _stext is page-aligned AND (3) _stext is not at Vaddr
		symbols, err := ef.Symbols()
		if err != nil && err != elf.ErrNoSymbols {
			return nil, err
		}
		for _, s := range symbols {
			if s.Name == "_stext" {
				// The kernel may use _stext as the mapping start address.
				stextOffset = &s.Value
				break
			}
		}
	}

	// Check that we can compute a base for the binary. This may not be the
	// correct base value, so we don't save it. We delay computing the actual base
	// value until we have a sample address for this mapping, so that we can
	// correctly identify the associated program segment that is needed to compute
	// the base.
	if _, err := elfexec.GetBase(&ef.FileHeader, elfexec.FindTextProgHeader(ef), stextOffset, start, limit, offset); err != nil {
		return nil, fmt.Errorf("could not identify base for %s: %v", name, err)
	}

	if b.fast || (!b.addr2lineFound && !b.llvmSymbolizerFound) {
		return &fileNM{file: file{
			b:       b,
			name:    name,
			buildID: buildID,
			m:       &elfMapping{start: start, limit: limit, offset: offset, stextOffset: stextOffset},
		}}, nil
	}
	return &fileAddr2Line{file: file{
		b:       b,
		name:    name,
		buildID: buildID,
		m:       &elfMapping{start: start, limit: limit, offset: offset, stextOffset: stextOffset},
	}}, nil
}

func (b *binrep) openPE(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	pf, err := pe.Open(name)
	if err != nil {
		return nil, fmt.Errorf("error parsing %s: %v", name, err)
	}
	defer pf.Close()

	var imageBase uint64
	switch h := pf.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		imageBase = uint64(h.ImageBase)
	case *pe.OptionalHeader64:
		imageBase = uint64(h.ImageBase)
	default:
		return nil, fmt.Errorf("unknown OptionalHeader %T", pf.OptionalHeader)
	}

	var base uint64
	if start > 0 {
		base = start - imageBase
	}
	if b.fast || (!b.addr2lineFound && !b.llvmSymbolizerFound) {
		return &fileNM{file: file{b: b, name: name, base: base}}, nil
	}
	return &fileAddr2Line{file: file{b: b, name: name, base: base}}, nil
}

// elfMapping stores the parameters of a runtime mapping that are needed to
// identify the ELF segment associated with a mapping.
type elfMapping struct {
	// Runtime mapping parameters.
	start, limit, offset uint64
	// Offset of _stext symbol. Only defined for kernel images, nil otherwise.
	stextOffset *uint64
}

// findProgramHeader returns the program segment that matches the current
// mapping and the given address, or an error if it cannot find a unique program
// header.
func (m *elfMapping) findProgramHeader(ef *elf.File, addr uint64) (*elf.ProgHeader, error) {
	// For user space executables, we try to find the actual program segment that
	// is associated with the given mapping. Skip this search if limit <= start.
	// We cannot use just a check on the start address of the mapping to tell if
	// it's a kernel / .ko module mapping, because with quipper address remapping
	// enabled, the address would be in the lower half of the address space.

	if m.stextOffset != nil || m.start >= m.limit || m.limit >= (uint64(1)<<63) {
		// For the kernel, find the program segment that includes the .text section.
		return elfexec.FindTextProgHeader(ef), nil
	}

	// Fetch all the loadable segments.
	var phdrs []elf.ProgHeader
	for i := range ef.Progs {
		if ef.Progs[i].Type == elf.PT_LOAD {
			phdrs = append(phdrs, ef.Progs[i].ProgHeader)
		}
	}
	// Some ELF files don't contain any loadable program segments, e.g. .ko
	// kernel modules. It's not an error to have no header in such cases.
	if len(phdrs) == 0 {
		return nil, nil
	}
	// Get all program headers associated with the mapping.
	headers := elfexec.ProgramHeadersForMapping(phdrs, m.offset, m.limit-m.start)
	if len(headers) == 0 {
		return nil, errors.New("no program header matches mapping info")
	}
	if len(headers) == 1 {
		return headers[0], nil
	}

	// Use the file offset corresponding to the address to symbolize, to narrow
	// down the header.
	return elfexec.HeaderForFileOffset(headers, addr-m.start+m.offset)
}

// file implements the binutils.ObjFile interface.
type file struct {
	b       *binrep
	name    string
	buildID string

	baseOnce sync.Once // Ensures the base, baseErr and isData are computed once.
	base     uint64
	baseErr  error // Any eventual error while computing the base.
	isData   bool
	// Mapping information. Relevant only for ELF files, nil otherwise.
	m *elfMapping
}

// computeBase computes the relocation base for the given binary file only if
// the elfMapping field is set. It populates the base and isData fields and
// returns an error.
func (f *file) computeBase(addr uint64) error {
	if f == nil || f.m == nil {
		return nil
	}
	if addr < f.m.start || addr >= f.m.limit {
		return fmt.Errorf("specified address %x is outside the mapping range [%x, %x] for file %q", addr, f.m.start, f.m.limit, f.name)
	}
	ef, err := elfOpen(f.name)
	if err != nil {
		return fmt.Errorf("error parsing %s: %v", f.name, err)
	}
	defer ef.Close()

	ph, err := f.m.findProgramHeader(ef, addr)
	if err != nil {
		return fmt.Errorf("failed to find program header for file %q, ELF mapping %#v, address %x: %v", f.name, *f.m, addr, err)
	}

	base, err := elfexec.GetBase(&ef.FileHeader, ph, f.m.stextOffset, f.m.start, f.m.limit, f.m.offset)
	if err != nil {
		return err
	}
	f.base = base
	f.isData = ph != nil && ph.Flags&elf.PF_X == 0
	return nil
}

func (f *file) Name() string {
	return f.name
}

func (f *file) ObjAddr(addr uint64) (uint64, error) {
	f.baseOnce.Do(func() { f.baseErr = f.computeBase(addr) })
	if f.baseErr != nil {
		return 0, f.baseErr
	}
	return addr - f.base, nil
}

func (f *file) BuildID() string {
	return f.buildID
}

func (f *file) SourceLine(addr uint64) ([]plugin.Frame, error) {
	f.baseOnce.Do(func() { f.baseErr = f.computeBase(addr) })
	if f.baseErr != nil {
		return nil, f.baseErr
	}
	return nil, nil
}

func (f *file) Close() error {
	return nil
}

func (f *file) Symbols(r *regexp.Regexp, addr uint64) ([]*plugin.Sym, error) {
	// Get from nm a list of symbols sorted by address.
	cmd := exec.Command(f.b.nm, "-n", f.name)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%v: %v", cmd.Args, err)
	}

	return findSymbols(out, f.name, r, addr)
}

// fileNM implements the binutils.ObjFile interface, using 'nm' to map
// addresses to symbols (without file/line number information). It is
// faster than fileAddr2Line.
type fileNM struct {
	file
	addr2linernm *addr2LinerNM
}

func (f *fileNM) SourceLine(addr uint64) ([]plugin.Frame, error) {
	f.baseOnce.Do(func() { f.baseErr = f.computeBase(addr) })
	if f.baseErr != nil {
		return nil, f.baseErr
	}
	if f.addr2linernm == nil {
		addr2liner, err := newAddr2LinerNM(f.b.nm, f.name, f.base)
		if err != nil {
			return nil, err
		}
		f.addr2linernm = addr2liner
	}
	return f.addr2linernm.addrInfo(addr)
}

// fileAddr2Line implements the binutils.ObjFile interface, using
// llvm-symbolizer, if that's available, or addr2line to map addresses to
// symbols (with file/line number information). It can be slow for large
// binaries with debug information.
type fileAddr2Line struct {
	once sync.Once
	file
	addr2liner     *addr2Liner
	llvmSymbolizer *llvmSymbolizer
	isData         bool
}

func (f *fileAddr2Line) SourceLine(addr uint64) ([]plugin.Frame, error) {
	f.baseOnce.Do(func() { f.baseErr = f.computeBase(addr) })
	if f.baseErr != nil {
		return nil, f.baseErr
	}
	f.once.Do(f.init)
	if f.llvmSymbolizer != nil {
		return f.llvmSymbolizer.addrInfo(addr)
	}
	if f.addr2liner != nil {
		return f.addr2liner.addrInfo(addr)
	}
	return nil, fmt.Errorf("could not find local addr2liner")
}

func (f *fileAddr2Line) init() {
	if llvmSymbolizer, err := newLLVMSymbolizer(f.b.llvmSymbolizer, f.name, f.base, f.isData); err == nil {
		f.llvmSymbolizer = llvmSymbolizer
		return
	}

	if addr2liner, err := newAddr2Liner(f.b.addr2line, f.name, f.base); err == nil {
		f.addr2liner = addr2liner

		// When addr2line encounters some gcc compiled binaries, it
		// drops interesting parts of names in anonymous namespaces.
		// Fallback to NM for better function names.
		if nm, err := newAddr2LinerNM(f.b.nm, f.name, f.base); err == nil {
			f.addr2liner.nm = nm
		}
	}
}

func (f *fileAddr2Line) Close() error {
	if f.llvmSymbolizer != nil {
		f.llvmSymbolizer.rw.close()
		f.llvmSymbolizer = nil
	}
	if f.addr2liner != nil {
		f.addr2liner.rw.close()
		f.addr2liner = nil
	}
	return nil
}
