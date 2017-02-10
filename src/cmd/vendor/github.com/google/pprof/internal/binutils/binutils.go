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
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/google/pprof/internal/elfexec"
	"github.com/google/pprof/internal/plugin"
)

// A Binutils implements plugin.ObjTool by invoking the GNU binutils.
// SetConfig must be called before any of the other methods.
type Binutils struct {
	// Commands to invoke.
	llvmSymbolizer      string
	llvmSymbolizerFound bool
	addr2line           string
	addr2lineFound      bool
	nm                  string
	nmFound             bool
	objdump             string
	objdumpFound        bool

	// if fast, perform symbolization using nm (symbol names only),
	// instead of file-line detail from the slower addr2line.
	fast bool
}

// SetFastSymbolization sets a toggle that makes binutils use fast
// symbolization (using nm), which is much faster than addr2line but
// provides only symbol name information (no file/line).
func (b *Binutils) SetFastSymbolization(fast bool) {
	b.fast = fast
}

// SetTools processes the contents of the tools option. It
// expects a set of entries separated by commas; each entry is a pair
// of the form t:path, where cmd will be used to look only for the
// tool named t. If t is not specified, the path is searched for all
// tools.
func (b *Binutils) SetTools(config string) {
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
	b.llvmSymbolizer, b.llvmSymbolizerFound = findExe("llvm-symbolizer", append(paths["llvm-symbolizer"], defaultPath...))
	b.addr2line, b.addr2lineFound = findExe("addr2line", append(paths["addr2line"], defaultPath...))
	b.nm, b.nmFound = findExe("nm", append(paths["nm"], defaultPath...))
	b.objdump, b.objdumpFound = findExe("objdump", append(paths["objdump"], defaultPath...))
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
func (b *Binutils) Disasm(file string, start, end uint64) ([]plugin.Inst, error) {
	if b.addr2line == "" {
		// Update the command invocations if not initialized.
		b.SetTools("")
	}
	cmd := exec.Command(b.objdump, "-d", "-C", "--no-show-raw-insn", "-l",
		fmt.Sprintf("--start-address=%#x", start),
		fmt.Sprintf("--stop-address=%#x", end),
		file)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%v: %v", cmd.Args, err)
	}

	return disassemble(out)
}

// Open satisfies the plugin.ObjTool interface.
func (b *Binutils) Open(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	if b.addr2line == "" {
		// Update the command invocations if not initialized.
		b.SetTools("")
	}

	// Make sure file is a supported executable.
	// The pprof driver uses Open to sniff the difference
	// between an executable and a profile.
	// For now, only ELF is supported.
	// Could read the first few bytes of the file and
	// use a table of prefixes if we need to support other
	// systems at some point.

	if _, err := os.Stat(name); err != nil {
		// For testing, do not require file name to exist.
		if strings.Contains(b.addr2line, "testdata/") {
			return &fileAddr2Line{file: file{b: b, name: name}}, nil
		}
		return nil, err
	}

	if f, err := b.openELF(name, start, limit, offset); err == nil {
		return f, nil
	}
	if f, err := b.openMachO(name, start, limit, offset); err == nil {
		return f, nil
	}
	return nil, fmt.Errorf("unrecognized binary: %s", name)
}

func (b *Binutils) openMachO(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	of, err := macho.Open(name)
	if err != nil {
		return nil, fmt.Errorf("Parsing %s: %v", name, err)
	}
	defer of.Close()

	if b.fast || (!b.addr2lineFound && !b.llvmSymbolizerFound) {
		return &fileNM{file: file{b: b, name: name}}, nil
	}
	return &fileAddr2Line{file: file{b: b, name: name}}, nil
}

func (b *Binutils) openELF(name string, start, limit, offset uint64) (plugin.ObjFile, error) {
	ef, err := elf.Open(name)
	if err != nil {
		return nil, fmt.Errorf("Parsing %s: %v", name, err)
	}
	defer ef.Close()

	var stextOffset *uint64
	var pageAligned = func(addr uint64) bool { return addr%4096 == 0 }
	if strings.Contains(name, "vmlinux") || !pageAligned(start) || !pageAligned(limit) || !pageAligned(offset) {
		// Reading all Symbols is expensive, and we only rarely need it so
		// we don't want to do it every time. But if _stext happens to be
		// page-aligned but isn't the same as Vaddr, we would symbolize
		// wrong. So if the name the addresses aren't page aligned, or if
		// the name is "vmlinux" we read _stext. We can be wrong if: (1)
		// someone passes a kernel path that doesn't contain "vmlinux" AND
		// (2) _stext is page-aligned AND (3) _stext is not at Vaddr
		symbols, err := ef.Symbols()
		if err != nil {
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

	base, err := elfexec.GetBase(&ef.FileHeader, nil, stextOffset, start, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("Could not identify base for %s: %v", name, err)
	}

	buildID := ""
	if f, err := os.Open(name); err == nil {
		if id, err := elfexec.GetBuildID(f); err == nil {
			buildID = fmt.Sprintf("%x", id)
		}
	}
	if b.fast || (!b.addr2lineFound && !b.llvmSymbolizerFound) {
		return &fileNM{file: file{b, name, base, buildID}}, nil
	}
	return &fileAddr2Line{file: file{b, name, base, buildID}}, nil
}

// file implements the binutils.ObjFile interface.
type file struct {
	b       *Binutils
	name    string
	base    uint64
	buildID string
}

func (f *file) Name() string {
	return f.name
}

func (f *file) Base() uint64 {
	return f.base
}

func (f *file) BuildID() string {
	return f.buildID
}

func (f *file) SourceLine(addr uint64) ([]plugin.Frame, error) {
	return []plugin.Frame{}, nil
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
// 'addr2line' to map addresses to symbols (with file/line number
// information). It can be slow for large binaries with debug
// information.
type fileAddr2Line struct {
	file
	addr2liner     *addr2Liner
	llvmSymbolizer *llvmSymbolizer
}

func (f *fileAddr2Line) SourceLine(addr uint64) ([]plugin.Frame, error) {
	if f.llvmSymbolizer != nil {
		return f.llvmSymbolizer.addrInfo(addr)
	}
	if f.addr2liner != nil {
		return f.addr2liner.addrInfo(addr)
	}

	if llvmSymbolizer, err := newLLVMSymbolizer(f.b.llvmSymbolizer, f.name, f.base); err == nil {
		f.llvmSymbolizer = llvmSymbolizer
		return f.llvmSymbolizer.addrInfo(addr)
	}

	if addr2liner, err := newAddr2Liner(f.b.addr2line, f.name, f.base); err == nil {
		f.addr2liner = addr2liner

		// When addr2line encounters some gcc compiled binaries, it
		// drops interesting parts of names in anonymous namespaces.
		// Fallback to NM for better function names.
		if nm, err := newAddr2LinerNM(f.b.nm, f.name, f.base); err == nil {
			f.addr2liner.nm = nm
		}
		return f.addr2liner.addrInfo(addr)
	}

	return nil, fmt.Errorf("could not find local addr2liner")
}

func (f *fileAddr2Line) Close() error {
	if f.addr2liner != nil {
		f.addr2liner.rw.close()
		f.addr2liner = nil
	}
	return nil
}
