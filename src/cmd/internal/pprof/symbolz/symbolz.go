// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package symbolz symbolizes a profile using the output from the symbolz
// service.
package symbolz

import (
	"bytes"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"internal/pprof/profile"
)

var (
	symbolzRE = regexp.MustCompile(`(0x[[:xdigit:]]+)\s+(.*)`)
)

// Symbolize symbolizes profile p by parsing data returned by a
// symbolz handler. syms receives the symbolz query (hex addresses
// separated by '+') and returns the symbolz output in a string. It
// symbolizes all locations based on their addresses, regardless of
// mapping.
func Symbolize(source string, syms func(string, string) ([]byte, error), p *profile.Profile) error {
	if source = symbolz(source, p); source == "" {
		// If the source is not a recognizable URL, do nothing.
		return nil
	}

	// Construct query of addresses to symbolize.
	var a []string
	for _, l := range p.Location {
		if l.Address != 0 && len(l.Line) == 0 {
			a = append(a, fmt.Sprintf("%#x", l.Address))
		}
	}

	if len(a) == 0 {
		// No addresses to symbolize.
		return nil
	}
	lines := make(map[uint64]profile.Line)
	functions := make(map[string]*profile.Function)
	if b, err := syms(source, strings.Join(a, "+")); err == nil {
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
				addr, err := strconv.ParseUint(symbol[1], 0, 64)
				if err != nil {
					return fmt.Errorf("unexpected parse failure %s: %v", symbol[1], err)
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
	}

	for _, l := range p.Location {
		if line, ok := lines[l.Address]; ok {
			l.Line = []profile.Line{line}
			if l.Mapping != nil {
				l.Mapping.HasFunctions = true
			}
		}
	}

	return nil
}

// symbolz returns the corresponding symbolz source for a profile URL.
func symbolz(source string, p *profile.Profile) string {
	if url, err := url.Parse(source); err == nil && url.Host != "" {
		if last := strings.LastIndex(url.Path, "/"); last != -1 {
			if strings.HasSuffix(url.Path[:last], "pprof") {
				url.Path = url.Path[:last] + "/symbol"
			} else {
				url.Path = url.Path[:last] + "/symbolz"
			}
			return url.String()
		}
	}

	return ""
}
