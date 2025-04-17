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

package binutils

import (
	"bytes"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/google/pprof/internal/plugin"
	"github.com/ianlancetaylor/demangle"
)

var (
	nmOutputRE                = regexp.MustCompile(`^\s*([[:xdigit:]]+)\s+(.)\s+(.*)`)
	objdumpAsmOutputRE        = regexp.MustCompile(`^\s*([[:xdigit:]]+):\s+(.*)`)
	objdumpOutputFileLine     = regexp.MustCompile(`^;?\s?(.*):([0-9]+)`)
	objdumpOutputFunction     = regexp.MustCompile(`^;?\s?(\S.*)\(\):`)
	objdumpOutputFunctionLLVM = regexp.MustCompile(`^([[:xdigit:]]+)?\s?(.*):`)
)

func findSymbols(syms []byte, file string, r *regexp.Regexp, address uint64) ([]*plugin.Sym, error) {
	// Collect all symbols from the nm output, grouping names mapped to
	// the same address into a single symbol.

	// The symbols to return.
	var symbols []*plugin.Sym

	// The current group of symbol names, and the address they are all at.
	names, start := []string{}, uint64(0)

	buf := bytes.NewBuffer(syms)

	for {
		symAddr, name, err := nextSymbol(buf)
		if err == io.EOF {
			// Done. If there was an unfinished group, append it.
			if len(names) != 0 {
				if match := matchSymbol(names, start, symAddr-1, r, address); match != nil {
					symbols = append(symbols, &plugin.Sym{Name: match, File: file, Start: start, End: symAddr - 1})
				}
			}

			// And return the symbols.
			return symbols, nil
		}

		if err != nil {
			// There was some kind of serious error reading nm's output.
			return nil, err
		}

		// If this symbol is at the same address as the current group, add it to the group.
		if symAddr == start {
			names = append(names, name)
			continue
		}

		// Otherwise append the current group to the list of symbols.
		if match := matchSymbol(names, start, symAddr-1, r, address); match != nil {
			symbols = append(symbols, &plugin.Sym{Name: match, File: file, Start: start, End: symAddr - 1})
		}

		// And start a new group.
		names, start = []string{name}, symAddr
	}
}

// matchSymbol checks if a symbol is to be selected by checking its
// name to the regexp and optionally its address. It returns the name(s)
// to be used for the matched symbol, or nil if no match
func matchSymbol(names []string, start, end uint64, r *regexp.Regexp, address uint64) []string {
	if address != 0 && address >= start && address <= end {
		return names
	}
	for _, name := range names {
		if r == nil || r.MatchString(name) {
			return []string{name}
		}

		// Match all possible demangled versions of the name.
		for _, o := range [][]demangle.Option{
			{demangle.NoClones},
			{demangle.NoParams, demangle.NoEnclosingParams},
			{demangle.NoParams, demangle.NoEnclosingParams, demangle.NoTemplateParams},
		} {
			if demangled, err := demangle.ToString(name, o...); err == nil && r.MatchString(demangled) {
				return []string{demangled}
			}
		}
	}
	return nil
}

// disassemble parses the output of the objdump command and returns
// the assembly instructions in a slice.
func disassemble(asm []byte) ([]plugin.Inst, error) {
	buf := bytes.NewBuffer(asm)
	function, file, line := "", "", 0
	var assembly []plugin.Inst
	for {
		input, err := buf.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return nil, err
			}
			if input == "" {
				break
			}
		}
		input = strings.TrimSpace(input)

		if fields := objdumpAsmOutputRE.FindStringSubmatch(input); len(fields) == 3 {
			if address, err := strconv.ParseUint(fields[1], 16, 64); err == nil {
				assembly = append(assembly,
					plugin.Inst{
						Addr:     address,
						Text:     fields[2],
						Function: function,
						File:     file,
						Line:     line,
					})
				continue
			}
		}
		if fields := objdumpOutputFileLine.FindStringSubmatch(input); len(fields) == 3 {
			if l, err := strconv.ParseUint(fields[2], 10, 32); err == nil {
				file, line = fields[1], int(l)
			}
			continue
		}
		if fields := objdumpOutputFunction.FindStringSubmatch(input); len(fields) == 2 {
			function = fields[1]
			continue
		} else {
			if fields := objdumpOutputFunctionLLVM.FindStringSubmatch(input); len(fields) == 3 {
				function = fields[2]
				continue
			}
		}
		// Reset on unrecognized lines.
		function, file, line = "", "", 0
	}

	return assembly, nil
}

// nextSymbol parses the nm output to find the next symbol listed.
// Skips over any output it cannot recognize.
func nextSymbol(buf *bytes.Buffer) (uint64, string, error) {
	for {
		line, err := buf.ReadString('\n')
		if err != nil {
			if err != io.EOF || line == "" {
				return 0, "", err
			}
		}
		line = strings.TrimSpace(line)

		if fields := nmOutputRE.FindStringSubmatch(line); len(fields) == 4 {
			if address, err := strconv.ParseUint(fields[1], 16, 64); err == nil {
				return address, fields[3], nil
			}
		}
	}
}
