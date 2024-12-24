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

// Package symbolizer provides a routine to populate a profile with
// symbol, file and line number information. It relies on the
// addr2liner and demangle packages to do the actual work.
package symbolizer

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/symbolz"
	"github.com/google/pprof/profile"
	"github.com/ianlancetaylor/demangle"
)

// Symbolizer implements the plugin.Symbolize interface.
type Symbolizer struct {
	Obj       plugin.ObjTool
	UI        plugin.UI
	Transport http.RoundTripper
}

// test taps for dependency injection
var symbolzSymbolize = symbolz.Symbolize
var localSymbolize = doLocalSymbolize
var demangleFunction = Demangle

// Symbolize attempts to symbolize profile p. First uses binutils on
// local binaries; if the source is a URL it attempts to get any
// missed entries using symbolz.
func (s *Symbolizer) Symbolize(mode string, sources plugin.MappingSources, p *profile.Profile) error {
	remote, local, fast, force, demanglerMode := true, true, false, false, ""
	for _, o := range strings.Split(strings.ToLower(mode), ":") {
		switch o {
		case "":
			continue
		case "none", "no":
			return nil
		case "local":
			remote, local = false, true
		case "fastlocal":
			remote, local, fast = false, true, true
		case "remote":
			remote, local = true, false
		case "force":
			force = true
		default:
			switch d := strings.TrimPrefix(o, "demangle="); d {
			case "full", "none", "templates":
				demanglerMode = d
				force = true
				continue
			case "default":
				continue
			}
			s.UI.PrintErr("ignoring unrecognized symbolization option: " + mode)
			s.UI.PrintErr("expecting -symbolize=[local|fastlocal|remote|none][:force][:demangle=[none|full|templates|default]")
		}
	}

	var err error
	if local {
		// Symbolize locally using binutils.
		if err = localSymbolize(p, fast, force, s.Obj, s.UI); err != nil {
			s.UI.PrintErr("local symbolization: " + err.Error())
		}
	}
	if remote {
		post := func(source, post string) ([]byte, error) {
			return postURL(source, post, s.Transport)
		}
		if err = symbolzSymbolize(p, force, sources, post, s.UI); err != nil {
			return err // Ran out of options.
		}
	}

	demangleFunction(p, force, demanglerMode)
	return nil
}

// postURL issues a POST to a URL over HTTP.
func postURL(source, post string, tr http.RoundTripper) ([]byte, error) {
	client := &http.Client{
		Transport: tr,
	}
	resp, err := client.Post(source, "application/octet-stream", strings.NewReader(post))
	if err != nil {
		return nil, fmt.Errorf("http post %s: %v", source, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http post %s: %v", source, statusCodeError(resp))
	}
	return io.ReadAll(resp.Body)
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

// doLocalSymbolize adds symbol and line number information to all locations
// in a profile. mode enables some options to control
// symbolization.
func doLocalSymbolize(prof *profile.Profile, fast, force bool, obj plugin.ObjTool, ui plugin.UI) error {
	if fast {
		if bu, ok := obj.(*binutils.Binutils); ok {
			bu.SetFastSymbolization(true)
		}
	}

	functions := map[profile.Function]*profile.Function{}
	addFunction := func(f *profile.Function) *profile.Function {
		if fp := functions[*f]; fp != nil {
			return fp
		}
		functions[*f] = f
		f.ID = uint64(len(prof.Function)) + 1
		prof.Function = append(prof.Function, f)
		return f
	}

	missingBinaries := false
	mappingLocs := map[*profile.Mapping][]*profile.Location{}
	for _, l := range prof.Location {
		mappingLocs[l.Mapping] = append(mappingLocs[l.Mapping], l)
	}
	for midx, m := range prof.Mapping {
		locs := mappingLocs[m]
		if len(locs) == 0 {
			// The mapping is dangling and has no locations pointing to it.
			continue
		}
		// Do not attempt to re-symbolize a mapping that has already been symbolized.
		if !force && (m.HasFunctions || m.HasFilenames || m.HasLineNumbers) {
			continue
		}
		if m.File == "" {
			if midx == 0 {
				ui.PrintErr("Main binary filename not available.")
				continue
			}
			missingBinaries = true
			continue
		}
		if m.Unsymbolizable() {
			// Skip well-known system mappings
			continue
		}
		if m.BuildID == "" {
			if u, err := url.Parse(m.File); err == nil && u.IsAbs() && strings.Contains(strings.ToLower(u.Scheme), "http") {
				// Skip mappings pointing to a source URL
				continue
			}
		}

		name := filepath.Base(m.File)
		if m.BuildID != "" {
			name += fmt.Sprintf(" (build ID %s)", m.BuildID)
		}
		f, err := obj.Open(m.File, m.Start, m.Limit, m.Offset, m.KernelRelocationSymbol)
		if err != nil {
			ui.PrintErr("Local symbolization failed for ", name, ": ", err)
			missingBinaries = true
			continue
		}
		if fid := f.BuildID(); m.BuildID != "" && fid != "" && fid != m.BuildID {
			ui.PrintErr("Local symbolization failed for ", name, ": build ID mismatch")
			f.Close()
			continue
		}
		symbolizeOneMapping(m, locs, f, addFunction)
		f.Close()
	}

	if missingBinaries {
		ui.PrintErr("Some binary filenames not available. Symbolization may be incomplete.\n" +
			"Try setting PPROF_BINARY_PATH to the search path for local binaries.")
	}
	return nil
}

func symbolizeOneMapping(m *profile.Mapping, locs []*profile.Location, obj plugin.ObjFile, addFunction func(*profile.Function) *profile.Function) {
	for _, l := range locs {
		stack, err := obj.SourceLine(l.Address)
		if err != nil || len(stack) == 0 {
			// No answers from addr2line.
			continue
		}

		l.Line = make([]profile.Line, len(stack))
		l.IsFolded = false
		for i, frame := range stack {
			if frame.Func != "" {
				m.HasFunctions = true
			}
			if frame.File != "" {
				m.HasFilenames = true
			}
			if frame.Line != 0 {
				m.HasLineNumbers = true
			}
			f := addFunction(&profile.Function{
				Name:       frame.Func,
				SystemName: frame.Func,
				Filename:   frame.File,
				StartLine:  int64(frame.StartLine),
			})
			l.Line[i] = profile.Line{
				Function: f,
				Line:     int64(frame.Line),
				Column:   int64(frame.Column),
			}
		}

		if len(stack) > 0 {
			m.HasInlineFrames = true
		}
	}
}

// Demangle updates the function names in a profile with demangled C++
// names, simplified according to demanglerMode. If force is set,
// overwrite any names that appear already demangled.
func Demangle(prof *profile.Profile, force bool, demanglerMode string) {
	if force {
		// Remove the current demangled names to force demangling
		for _, f := range prof.Function {
			if f.Name != "" && f.SystemName != "" {
				f.Name = f.SystemName
			}
		}
	}

	options := demanglerModeToOptions(demanglerMode)
	for _, fn := range prof.Function {
		demangleSingleFunction(fn, options)
	}
}

func demanglerModeToOptions(demanglerMode string) []demangle.Option {
	switch demanglerMode {
	case "": // demangled, simplified: no parameters, no templates, no return type
		return []demangle.Option{demangle.NoParams, demangle.NoEnclosingParams, demangle.NoTemplateParams}
	case "templates": // demangled, simplified: no parameters, no return type
		return []demangle.Option{demangle.NoParams, demangle.NoEnclosingParams}
	case "full":
		return []demangle.Option{demangle.NoClones}
	case "none": // no demangling
		return []demangle.Option{}
	}

	panic(fmt.Sprintf("unknown demanglerMode %s", demanglerMode))
}

func demangleSingleFunction(fn *profile.Function, options []demangle.Option) {
	if fn.Name != "" && fn.SystemName != fn.Name {
		return // Already demangled.
	}
	// Copy the options because they may be updated by the call.
	o := make([]demangle.Option, len(options))
	copy(o, options)
	if demangled := demangle.Filter(fn.SystemName, o...); demangled != fn.SystemName {
		fn.Name = demangled
		return
	}
	// Could not demangle. Apply heuristics in case the name is
	// already demangled.
	name := fn.SystemName
	if looksLikeDemangledCPlusPlus(name) {
		for _, o := range options {
			switch o {
			case demangle.NoParams:
				name = removeMatching(name, '(', ')')
			case demangle.NoTemplateParams:
				name = removeMatching(name, '<', '>')
			}
		}
	}
	fn.Name = name
}

// looksLikeDemangledCPlusPlus is a heuristic to decide if a name is
// the result of demangling C++. If so, further heuristics will be
// applied to simplify the name.
func looksLikeDemangledCPlusPlus(demangled string) bool {
	// Skip java names of the form "class.<init>".
	if strings.Contains(demangled, ".<") {
		return false
	}
	// Skip Go names of the form "foo.(*Bar[...]).Method".
	if strings.Contains(demangled, "]).") {
		return false
	}
	return strings.ContainsAny(demangled, "<>[]") || strings.Contains(demangled, "::")
}

// removeMatching removes nested instances of start..end from name.
func removeMatching(name string, start, end byte) string {
	s := string(start) + string(end)
	var nesting, first, current int
	for index := strings.IndexAny(name[current:], s); index != -1; index = strings.IndexAny(name[current:], s) {
		switch current += index; name[current] {
		case start:
			nesting++
			if nesting == 1 {
				first = current
			}
		case end:
			nesting--
			switch {
			case nesting < 0:
				return name // Mismatch, abort
			case nesting == 0:
				name = name[:first] + name[current+1:]
				current = first - 1
			}
		}
		current++
	}
	return name
}
