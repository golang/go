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

	"github.com/google/pprof/internal/plugin"
	"github.com/ianlancetaylor/demangle"
)

var (
	nmOutputRE            = regexp.MustCompile(`^\s*([[:xdigit:]]+)\s+(.)\s+(.*)`)
	objdumpAsmOutputRE    = regexp.MustCompile(`^\s*([[:xdigit:]]+):\s+(.*)`)
	objdumpOutputFileLine = regexp.MustCompile(`^(.*):([0-9]+)`)
	objdumpOutputFunction = regexp.MustCompile(`^(\S.*)\(\):`)
)

func findSymbols(syms []byte, file string, r *regexp.Regexp, address uint64) ([]*plugin.Sym, error) {
	// Collect all symbols from the nm output, grouping names mapped to
	// the same address into a single symbol.
	var symbols []*plugin.Sym
	names, start := []string{}, uint64(0)
	buf := bytes.NewBuffer(syms)
	for symAddr, name, err := nextSymbol(buf); err == nil; symAddr, name, err = nextSymbol(buf) {
		if err != nil {
			return nil, err
		}
		if start == symAddr {
			names = append(names, name)
			continue
		}
		if match := matchSymbol(names, start, symAddr-1, r, address); match != nil {
			symbols = append(symbols, &plugin.Sym{Name: match, File: file, Start: start, End: symAddr - 1})
		}
		names, start = []string{name}, symAddr
	}

	return symbols, nil
}

// matchSymbol checks if a symbol is to be selected by checking its
// name to the regexp and optionally its address. It returns the name(s)
// to be used for the matched symbol, or nil if no match
func matchSymbol(names []string, start, end uint64, r *regexp.Regexp, address uint64) []string {
	if address != 0 && address >= start && address <= end {
		return names
	}
	for _, name := range names {
		if r.MatchString(name) {
			return []string{name}
		}

		// Match all possible demangled versions of the name.
		for _, o := range [][]demangle.Option{
			{demangle.NoClones},
			{demangle.NoParams},
			{demangle.NoParams, demangle.NoTemplateParams},
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

		if fields := nmOutputRE.FindStringSubmatch(line); len(fields) == 4 {
			if address, err := strconv.ParseUint(fields[1], 16, 64); err == nil {
				return address, fields[3], nil
			}
		}
	}
}
