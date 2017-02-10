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

// addr2LinerNM is a connection to an nm command for obtaining address
// information from a binary.
type addr2LinerNM struct {
	m []symbolInfo // Sorted list of addresses from binary.
}

type symbolInfo struct {
	address uint64
	name    string
}

//  newAddr2LinerNM starts the given nm command reporting information about the
// given executable file. If file is a shared library, base should be
// the address at which it was mapped in the program under
// consideration.
func newAddr2LinerNM(cmd, file string, base uint64) (*addr2LinerNM, error) {
	if cmd == "" {
		cmd = defaultNM
	}

	a := &addr2LinerNM{
		m: []symbolInfo{},
	}

	var b bytes.Buffer
	c := exec.Command(cmd, "-n", file)
	c.Stdout = &b

	if err := c.Run(); err != nil {
		return nil, err
	}

	// Parse nm output and populate symbol map.
	// Skip lines we fail to parse.
	buf := bufio.NewReader(&b)
	for {
		line, err := buf.ReadString('\n')
		if line == "" && err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		line = strings.TrimSpace(line)
		fields := strings.SplitN(line, " ", 3)
		if len(fields) != 3 {
			continue
		}
		address, err := strconv.ParseUint(fields[0], 16, 64)
		if err != nil {
			continue
		}
		a.m = append(a.m, symbolInfo{
			address: address + base,
			name:    fields[2],
		})
	}

	return a, nil
}

// addrInfo returns the stack frame information for a specific program
// address. It returns nil if the address could not be identified.
func (a *addr2LinerNM) addrInfo(addr uint64) ([]plugin.Frame, error) {
	if len(a.m) == 0 || addr < a.m[0].address || addr > a.m[len(a.m)-1].address {
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

	// Address is between a.m[low] and a.m[high].
	// Pick low, as it represents [low, high).
	f := []plugin.Frame{
		{
			Func: a.m[low].name,
		},
	}
	return f, nil
}
