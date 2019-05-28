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

// Package symbolz symbolizes a profile using the output from the symbolz
// service.
package symbolz

import (
	"bytes"
	"fmt"
	"io"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/profile"
)

var (
	symbolzRE = regexp.MustCompile(`(0x[[:xdigit:]]+)\s+(.*)`)
)

// Symbolize symbolizes profile p by parsing data returned by a symbolz
// handler. syms receives the symbolz query (hex addresses separated by '+')
// and returns the symbolz output in a string. If force is false, it will only
// symbolize locations from mappings not already marked as HasFunctions. Never
// attempts symbolization of addresses from unsymbolizable system
// mappings as those may look negative - e.g. "[vsyscall]".
func Symbolize(p *profile.Profile, force bool, sources plugin.MappingSources, syms func(string, string) ([]byte, error), ui plugin.UI) error {
	for _, m := range p.Mapping {
		if !force && m.HasFunctions {
			// Only check for HasFunctions as symbolz only populates function names.
			continue
		}
		// Skip well-known system mappings.
		if m.Unsymbolizable() {
			continue
		}
		mappingSources := sources[m.File]
		if m.BuildID != "" {
			mappingSources = append(mappingSources, sources[m.BuildID]...)
		}
		for _, source := range mappingSources {
			if symz := symbolz(source.Source); symz != "" {
				if err := symbolizeMapping(symz, int64(source.Start)-int64(m.Start), syms, m, p); err != nil {
					return err
				}
				m.HasFunctions = true
				break
			}
		}
	}

	return nil
}

// hasGperftoolsSuffix checks whether path ends with one of the suffixes listed in
// pprof_remote_servers.html from the gperftools distribution
func hasGperftoolsSuffix(path string) bool {
	suffixes := []string{
		"/pprof/heap",
		"/pprof/growth",
		"/pprof/profile",
		"/pprof/pmuprofile",
		"/pprof/contention",
	}
	for _, s := range suffixes {
		if strings.HasSuffix(path, s) {
			return true
		}
	}
	return false
}

// symbolz returns the corresponding symbolz source for a profile URL.
func symbolz(source string) string {
	if url, err := url.Parse(source); err == nil && url.Host != "" {
		// All paths in the net/http/pprof Go package contain /debug/pprof/
		if strings.Contains(url.Path, "/debug/pprof/") || hasGperftoolsSuffix(url.Path) {
			url.Path = path.Clean(url.Path + "/../symbol")
		} else {
			url.Path = "/symbolz"
		}
		url.RawQuery = ""
		return url.String()
	}

	return ""
}

// symbolizeMapping symbolizes locations belonging to a Mapping by querying
// a symbolz handler. An offset is applied to all addresses to take care of
// normalization occurred for merged Mappings.
func symbolizeMapping(source string, offset int64, syms func(string, string) ([]byte, error), m *profile.Mapping, p *profile.Profile) error {
	// Construct query of addresses to symbolize.
	var a []string
	for _, l := range p.Location {
		if l.Mapping == m && l.Address != 0 && len(l.Line) == 0 {
			// Compensate for normalization.
			addr, overflow := adjust(l.Address, offset)
			if overflow {
				return fmt.Errorf("cannot adjust address %d by %d, it would overflow (mapping %v)", l.Address, offset, l.Mapping)
			}
			a = append(a, fmt.Sprintf("%#x", addr))
		}
	}

	if len(a) == 0 {
		// No addresses to symbolize.
		return nil
	}

	lines := make(map[uint64]profile.Line)
	functions := make(map[string]*profile.Function)

	b, err := syms(source, strings.Join(a, "+"))
	if err != nil {
		return err
	}

	buf := bytes.NewBuffer(b)
	for {
		l, err := buf.ReadString('\n')

		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if symbol := symbolzRE.FindStringSubmatch(l); len(symbol) == 3 {
			origAddr, err := strconv.ParseUint(symbol[1], 0, 64)
			if err != nil {
				return fmt.Errorf("unexpected parse failure %s: %v", symbol[1], err)
			}
			// Reapply offset expected by the profile.
			addr, overflow := adjust(origAddr, -offset)
			if overflow {
				return fmt.Errorf("cannot adjust symbolz address %d by %d, it would overflow", origAddr, -offset)
			}

			name := symbol[2]
			fn := functions[name]
			if fn == nil {
				fn = &profile.Function{
					ID:         uint64(len(p.Function) + 1),
					Name:       name,
					SystemName: name,
				}
				functions[name] = fn
				p.Function = append(p.Function, fn)
			}

			lines[addr] = profile.Line{Function: fn}
		}
	}

	for _, l := range p.Location {
		if l.Mapping != m {
			continue
		}
		if line, ok := lines[l.Address]; ok {
			l.Line = []profile.Line{line}
		}
	}

	return nil
}

// adjust shifts the specified address by the signed offset. It returns the
// adjusted address. It signals that the address cannot be adjusted without an
// overflow by returning true in the second return value.
func adjust(addr uint64, offset int64) (uint64, bool) {
	adj := uint64(int64(addr) + offset)
	if offset < 0 {
		if adj >= addr {
			return 0, true
		}
	} else {
		if adj < addr {
			return 0, true
		}
	}
	return adj, false
}
