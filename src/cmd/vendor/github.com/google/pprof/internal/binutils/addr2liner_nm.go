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
	"bufio"
	"bytes"
	"io"
	"os/exec"
	"strconv"
	"strings"

	"github.com/google/pprof/internal/plugin"
)

const (
	defaultNM = "nm"
)

// addr2LinerNM is a connection to an nm command for obtaining symbol
// information from a binary.
type addr2LinerNM struct {
	m []symbolInfo // Sorted list of symbol addresses from binary.
}

type symbolInfo struct {
	address uint64
	size    uint64
	name    string
	symType string
}

// isData returns if the symbol has a known data object symbol type.
func (s *symbolInfo) isData() bool {
	// The following symbol types are taken from https://linux.die.net/man/1/nm:
	// Lowercase letter means local symbol, uppercase denotes a global symbol.
	// - b or B: the symbol is in the uninitialized data section, e.g. .bss;
	// - d or D: the symbol is in the initialized data section;
	// - r or R: the symbol is in a read only data section;
	// - v or V: the symbol is a weak object;
	// - W: the symbol is a weak symbol that has not been specifically tagged as a
	//      weak object symbol. Experiments with some binaries, showed these to be
	//      mostly data objects.
	return strings.ContainsAny(s.symType, "bBdDrRvVW")
}

// newAddr2LinerNM starts the given nm command reporting information about the
// given executable file. If file is a shared library, base should be the
// address at which it was mapped in the program under consideration.
func newAddr2LinerNM(cmd, file string, base uint64) (*addr2LinerNM, error) {
	if cmd == "" {
		cmd = defaultNM
	}
	var b bytes.Buffer
	c := exec.Command(cmd, "--numeric-sort", "--print-size", "--format=posix", file)
	c.Stdout = &b
	if err := c.Run(); err != nil {
		return nil, err
	}
	return parseAddr2LinerNM(base, &b)
}

func parseAddr2LinerNM(base uint64, nm io.Reader) (*addr2LinerNM, error) {
	a := &addr2LinerNM{
		m: []symbolInfo{},
	}

	// Parse nm output and populate symbol map.
	// Skip lines we fail to parse.
	buf := bufio.NewReader(nm)
	for {
		line, err := buf.ReadString('\n')
		if line == "" && err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		line = strings.TrimSpace(line)
		fields := strings.Split(line, " ")
		if len(fields) != 4 {
			continue
		}
		address, err := strconv.ParseUint(fields[2], 16, 64)
		if err != nil {
			continue
		}
		size, err := strconv.ParseUint(fields[3], 16, 64)
		if err != nil {
			continue
		}
		a.m = append(a.m, symbolInfo{
			address: address + base,
			size:    size,
			name:    fields[0],
			symType: fields[1],
		})
	}

	return a, nil
}

// addrInfo returns the stack frame information for a specific program
// address. It returns nil if the address could not be identified.
func (a *addr2LinerNM) addrInfo(addr uint64) ([]plugin.Frame, error) {
	if len(a.m) == 0 || addr < a.m[0].address || addr >= (a.m[len(a.m)-1].address+a.m[len(a.m)-1].size) {
		return nil, nil
	}

	// Binary search. Search until low, high are separated by 1.
	low, high := 0, len(a.m)
	for low+1 < high {
		mid := (low + high) / 2
		v := a.m[mid].address
		if addr == v {
			low = mid
			break
		} else if addr > v {
			low = mid
		} else {
			high = mid
		}
	}

	// Address is between a.m[low] and a.m[high]. Pick low, as it represents
	// [low, high). For data symbols, we use a strict check that the address is in
	// the [start, start + size) range of a.m[low].
	if a.m[low].isData() && addr >= (a.m[low].address+a.m[low].size) {
		return nil, nil
	}
	return []plugin.Frame{{Func: a.m[low].name}}, nil
}
