// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// pprof is a tool for visualization of profile.data. It is based on
// the upstream version at github.com/google/pprof, with minor
// modifications specific to the Go distribution. Please consider
// upstreaming any modifications to these packages.

package main

import (
	"crypto/tls"
	"debug/dwarf"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/internal/objfile"
	"cmd/internal/telemetry"

	"github.com/google/pprof/driver"
	"github.com/google/pprof/profile"
)

func main() {
	telemetry.Start()
	telemetry.Inc("pprof/invocations")
	options := &driver.Options{
		Fetch: new(fetcher),
		Obj:   new(objTool),
		UI:    newUI(),
	}
	err := driver.PProf(options)
	telemetry.CountFlags("pprof/flag:", *flag.CommandLine) // pprof will use the flag package as its default
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(2)
	}
}

type fetcher struct {
}

func (f *fetcher) Fetch(src string, duration, timeout time.Duration) (*profile.Profile, string, error) {
	// Firstly, determine if the src is an existing file on the disk.
	// If it is a file, let regular pprof open it.
	// If it is not a file, when the src contains `:`
	// (e.g. mem_2023-11-02_03:55:24 or abc:123/mem_2023-11-02_03:55:24),
	// url.Parse will recognize it as a link and ultimately report an error,
	// similar to `abc:123/mem_2023-11-02_03:55:24:
	// Get "http://abc:123/mem_2023-11-02_03:55:24": dial tcp: lookup abc: no such host`
	if _, openErr := os.Stat(src); openErr == nil {
		return nil, "", nil
	}
	sourceURL, timeout := adjustURL(src, duration, timeout)
	if sourceURL == "" {
		// Could not recognize URL, let regular pprof attempt to fetch the profile (eg. from a file)
		return nil, "", nil
	}
	fmt.Fprintln(os.Stderr, "Fetching profile over HTTP from", sourceURL)
	if duration > 0 {
		fmt.Fprintf(os.Stderr, "Please wait... (%v)\n", duration)
	}
	p, err := getProfile(sourceURL, timeout)
	return p, sourceURL, err
}

func getProfile(source string, timeout time.Duration) (*profile.Profile, error) {
	url, err := url.Parse(source)
	if err != nil {
		return nil, err
	}

	var tlsConfig *tls.Config
	if url.Scheme == "https+insecure" {
		tlsConfig = &tls.Config{
			InsecureSkipVerify: true,
		}
		url.Scheme = "https"
		source = url.String()
	}

	client := &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: timeout + 5*time.Second,
			Proxy:                 http.ProxyFromEnvironment,
			TLSClientConfig:       tlsConfig,
		},
	}
	resp, err := client.Get(source)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, statusCodeError(resp)
	}
	return profile.Parse(resp.Body)
}

func statusCodeError(resp *http.Response) error {
	if resp.Header.Get("X-Go-Pprof") != "" && strings.Contains(resp.Header.Get("Content-Type"), "text/plain") {
		// error is from pprof endpoint
		if body, err := io.ReadAll(resp.Body); err == nil {
			return fmt.Errorf("server response: %s - %s", resp.Status, body)
		}
	}
	return fmt.Errorf("server response: %s", resp.Status)
}

// cpuProfileHandler is the Go pprof CPU profile handler URL.
const cpuProfileHandler = "/debug/pprof/profile"

// adjustURL applies the duration/timeout values and Go specific defaults.
func adjustURL(source string, duration, timeout time.Duration) (string, time.Duration) {
	u, err := url.Parse(source)
	if err != nil || (u.Host == "" && u.Scheme != "" && u.Scheme != "file") {
		// Try adding http:// to catch sources of the form hostname:port/path.
		// url.Parse treats "hostname" as the scheme.
		u, err = url.Parse("http://" + source)
	}
	if err != nil || u.Host == "" {
		return "", 0
	}

	if u.Path == "" || u.Path == "/" {
		u.Path = cpuProfileHandler
	}

	// Apply duration/timeout overrides to URL.
	values := u.Query()
	if duration > 0 {
		values.Set("seconds", fmt.Sprint(int(duration.Seconds())))
	} else {
		if urlSeconds := values.Get("seconds"); urlSeconds != "" {
			if us, err := strconv.ParseInt(urlSeconds, 10, 32); err == nil {
				duration = time.Duration(us) * time.Second
			}
		}
	}
	if timeout <= 0 {
		if duration > 0 {
			timeout = duration + duration/2
		} else {
			timeout = 60 * time.Second
		}
	}
	u.RawQuery = values.Encode()
	return u.String(), timeout
}

// objTool implements driver.ObjTool using Go libraries
// (instead of invoking GNU binutils).
type objTool struct {
	mu          sync.Mutex
	disasmCache map[string]*objfile.Disasm
}

func (*objTool) Open(name string, start, limit, offset uint64, relocationSymbol string) (driver.ObjFile, error) {
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

func (t *objTool) Disasm(file string, start, end uint64, intelSyntax bool) ([]driver.Inst, error) {
	if intelSyntax {
		return nil, fmt.Errorf("printing assembly in Intel syntax is not supported")
	}
	d, err := t.cachedDisasm(file)
	if err != nil {
		return nil, err
	}
	var asm []driver.Inst
	d.Decode(start, end, nil, false, func(pc, size uint64, file string, line int, text string) {
		asm = append(asm, driver.Inst{Addr: pc, File: file, Line: line, Text: text})
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

// file implements driver.ObjFile using Go libraries
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

func (f *file) ObjAddr(addr uint64) (uint64, error) {
	return addr - f.offset, nil
}

func (f *file) BuildID() string {
	// No support for build ID.
	return ""
}

func (f *file) SourceLine(addr uint64) ([]driver.Frame, error) {
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
		frame := []driver.Frame{
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
func (f *file) dwarfSourceLine(addr uint64) []driver.Frame {
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
func (f *file) dwarfSourceLineEntry(r *dwarf.Reader, entry *dwarf.Entry, addr uint64) []driver.Frame {
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

	frames := []driver.Frame{
		{
			Func: name,
			File: lentry.File.Name,
			Line: lentry.Line,
		},
	}

	return frames
}

func (f *file) Symbols(r *regexp.Regexp, addr uint64) ([]*driver.Sym, error) {
	if f.sym == nil {
		sym, err := f.file.Symbols()
		if err != nil {
			return nil, err
		}
		f.sym = sym
	}
	var out []*driver.Sym
	for _, s := range f.sym {
		// Ignore a symbol with address 0 and size 0.
		// An ELF STT_FILE symbol will look like that.
		if s.Addr == 0 && s.Size == 0 {
			continue
		}
		if (r == nil || r.MatchString(s.Name)) && (addr == 0 || s.Addr <= addr && addr < s.Addr+uint64(s.Size)) {
			out = append(out, &driver.Sym{
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

// newUI will be set in readlineui.go in some platforms
// for interactive readline functionality.
var newUI = func() driver.UI { return nil }
