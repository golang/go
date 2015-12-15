// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"debug/gosym"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"

	"cmd/internal/objfile"
	"cmd/pprof/internal/commands"
	"cmd/pprof/internal/driver"
	"cmd/pprof/internal/fetch"
	"cmd/pprof/internal/plugin"
	"cmd/pprof/internal/profile"
	"cmd/pprof/internal/symbolizer"
	"cmd/pprof/internal/symbolz"
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
			// Ignore these options, -force is recognized by symbolizer.Symbolize
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
	d.Decode(start, end, func(pc, size uint64, file string, line int, text string) {
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
	name string
	sym  []objfile.Sym
	file *objfile.File
	pcln *gosym.Table
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
	file, line, fn := f.pcln.PCToLine(addr)
	if fn == nil {
		return nil, fmt.Errorf("no line information for PC=%#x", addr)
	}
	frame := []plugin.Frame{
		{
			Func: fn.Name,
			File: file,
			Line: line,
		},
	}
	return frame, nil
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
