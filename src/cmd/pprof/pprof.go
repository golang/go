// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"debug/dwarf"
	"flag"
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"

	"cmd/internal/objfile"
	"cmd/pprof/internal/commands"
	"cmd/pprof/internal/driver"
	"cmd/pprof/internal/fetch"
	"cmd/pprof/internal/plugin"
	"cmd/pprof/internal/symbolizer"
	"cmd/pprof/internal/symbolz"
	"internal/pprof/profile"
)

func main() {
	var extraCommands map[string]*commands.Command // no added Go-specific commands
	if err := driver.PProf(flags{}, fetch.Fetcher, symbolize, new(objTool), plugin.StandardUI(), extraCommands); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(2)
	}
}

// symbolize attempts to symbolize profile p.
// If the source is a local binary, it tries using symbolizer and obj.
// If the source is a URL, it fetches symbol information using symbolz.
func symbolize(mode, source string, p *profile.Profile, obj plugin.ObjTool, ui plugin.UI) error {
	remote, local := true, true
	for _, o := range strings.Split(strings.ToLower(mode), ":") {
		switch o {
		case "none", "no":
			return nil
		case "local":
			remote, local = false, true
		case "remote":
			remote, local = true, false
		default:
			ui.PrintErr("ignoring unrecognized symbolization option: " + mode)
			ui.PrintErr("expecting -symbolize=[local|remote|none][:force]")
			fallthrough
		case "", "force":
			// -force is recognized by symbolizer.Symbolize.
			// If the source is remote, and the mapping file
			// does not exist, don't use local symbolization.
			if isRemote(source) {
				if len(p.Mapping) == 0 {
					local = false
				} else if _, err := os.Stat(p.Mapping[0].File); err != nil {
					local = false
				}
			}
		}
	}

	var err error
	if local {
		// Symbolize using binutils.
		if err = symbolizer.Symbolize(mode, p, obj, ui); err == nil {
			return nil
		}
	}
	if remote {
		err = symbolz.Symbolize(source, fetch.PostURL, p)
	}
	return err
}

// isRemote returns whether source is a URL for a remote source.
func isRemote(source string) bool {
	url, err := url.Parse(source)
	if err != nil {
		url, err = url.Parse("http://" + source)
		if err != nil {
			return false
		}
	}
	if scheme := strings.ToLower(url.Scheme); scheme == "" || scheme == "file" {
		return false
	}
	return true
}

// flags implements the driver.FlagPackage interface using the builtin flag package.
type flags struct {
}

func (flags) Bool(o string, d bool, c string) *bool {
	return flag.Bool(o, d, c)
}

func (flags) Int(o string, d int, c string) *int {
	return flag.Int(o, d, c)
}

func (flags) Float64(o string, d float64, c string) *float64 {
	return flag.Float64(o, d, c)
}

func (flags) String(o, d, c string) *string {
	return flag.String(o, d, c)
}

func (flags) Parse(usage func()) []string {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		usage()
	}
	return args
}

func (flags) ExtraUsage() string {
	return ""
}

// objTool implements plugin.ObjTool using Go libraries
// (instead of invoking GNU binutils).
type objTool struct {
	mu          sync.Mutex
	disasmCache map[string]*objfile.Disasm
}

func (*objTool) Open(name string, start uint64) (plugin.ObjFile, error) {
	of, err := objfile.Open(name)
	if err != nil {
		return nil, err
	}
	f := &file{
		name: name,
		file: of,
	}
	if start != 0 {
		if load, err := of.LoadAddress(); err == nil {
			f.offset = start - load
		}
	}
	return f, nil
}

func (*objTool) Demangle(names []string) (map[string]string, error) {
	// No C++, nothing to demangle.
	return make(map[string]string), nil
}

func (t *objTool) Disasm(file string, start, end uint64) ([]plugin.Inst, error) {
	d, err := t.cachedDisasm(file)
	if err != nil {
		return nil, err
	}
	var asm []plugin.Inst
	d.Decode(start, end, nil, func(pc, size uint64, file string, line int, text string) {
		asm = append(asm, plugin.Inst{Addr: pc, File: file, Line: line, Text: text})
	})
	return asm, nil
}

func (t *objTool) cachedDisasm(file string) (*objfile.Disasm, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.disasmCache == nil {
		t.disasmCache = make(map[string]*objfile.Disasm)
	}
	d := t.disasmCache[file]
	if d != nil {
		return d, nil
	}
	f, err := objfile.Open(file)
	if err != nil {
		return nil, err
	}
	d, err = f.Disasm()
	f.Close()
	if err != nil {
		return nil, err
	}
	t.disasmCache[file] = d
	return d, nil
}

func (*objTool) SetConfig(config string) {
	// config is usually used to say what binaries to invoke.
	// Ignore entirely.
}

// file implements plugin.ObjFile using Go libraries
// (instead of invoking GNU binutils).
// A file represents a single executable being analyzed.
type file struct {
	name   string
	offset uint64
	sym    []objfile.Sym
	file   *objfile.File
	pcln   objfile.Liner

	triedDwarf bool
	dwarf      *dwarf.Data
}

func (f *file) Name() string {
	return f.name
}

func (f *file) Base() uint64 {
	// No support for shared libraries.
	return 0
}

func (f *file) BuildID() string {
	// No support for build ID.
	return ""
}

func (f *file) SourceLine(addr uint64) ([]plugin.Frame, error) {
	if f.pcln == nil {
		pcln, err := f.file.PCLineTable()
		if err != nil {
			return nil, err
		}
		f.pcln = pcln
	}
	addr -= f.offset
	file, line, fn := f.pcln.PCToLine(addr)
	if fn != nil {
		frame := []plugin.Frame{
			{
				Func: fn.Name,
				File: file,
				Line: line,
			},
		}
		return frame, nil
	}

	frames := f.dwarfSourceLine(addr)
	if frames != nil {
		return frames, nil
	}

	return nil, fmt.Errorf("no line information for PC=%#x", addr)
}

// dwarfSourceLine tries to get file/line information using DWARF.
// This is for C functions that appear in the profile.
// Returns nil if there is no information available.
func (f *file) dwarfSourceLine(addr uint64) []plugin.Frame {
	if f.dwarf == nil && !f.triedDwarf {
		// Ignore any error--we don't care exactly why there
		// is no DWARF info.
		f.dwarf, _ = f.file.DWARF()
		f.triedDwarf = true
	}

	if f.dwarf != nil {
		r := f.dwarf.Reader()
		unit, err := r.SeekPC(addr)
		if err == nil {
			if frames := f.dwarfSourceLineEntry(r, unit, addr); frames != nil {
				return frames
			}
		}
	}

	return nil
}

// dwarfSourceLineEntry tries to get file/line information from a
// DWARF compilation unit. Returns nil if it doesn't find anything.
func (f *file) dwarfSourceLineEntry(r *dwarf.Reader, entry *dwarf.Entry, addr uint64) []plugin.Frame {
	lines, err := f.dwarf.LineReader(entry)
	if err != nil {
		return nil
	}
	var lentry dwarf.LineEntry
	if err := lines.SeekPC(addr, &lentry); err != nil {
		return nil
	}

	// Try to find the function name.
	name := ""
FindName:
	for entry, err := r.Next(); entry != nil && err == nil; entry, err = r.Next() {
		if entry.Tag == dwarf.TagSubprogram {
			ranges, err := f.dwarf.Ranges(entry)
			if err != nil {
				return nil
			}
			for _, pcs := range ranges {
				if pcs[0] <= addr && addr < pcs[1] {
					var ok bool
					// TODO: AT_linkage_name, AT_MIPS_linkage_name.
					name, ok = entry.Val(dwarf.AttrName).(string)
					if ok {
						break FindName
					}
				}
			}
		}
	}

	// TODO: Report inlined functions.

	frames := []plugin.Frame{
		{
			Func: name,
			File: lentry.File.Name,
			Line: lentry.Line,
		},
	}

	return frames
}

func (f *file) Symbols(r *regexp.Regexp, addr uint64) ([]*plugin.Sym, error) {
	if f.sym == nil {
		sym, err := f.file.Symbols()
		if err != nil {
			return nil, err
		}
		f.sym = sym
	}
	var out []*plugin.Sym
	for _, s := range f.sym {
		// Ignore a symbol with address 0 and size 0.
		// An ELF STT_FILE symbol will look like that.
		if s.Addr == 0 && s.Size == 0 {
			continue
		}
		if (r == nil || r.MatchString(s.Name)) && (addr == 0 || s.Addr <= addr && addr < s.Addr+uint64(s.Size)) {
			out = append(out, &plugin.Sym{
				Name:  []string{s.Name},
				File:  f.name,
				Start: s.Addr,
				End:   s.Addr + uint64(s.Size) - 1,
			})
		}
	}
	return out, nil
}

func (f *file) Close() error {
	f.file.Close()
	return nil
}
