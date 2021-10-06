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
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/google/pprof/internal/plugin"
)

const (
	defaultLLVMSymbolizer = "llvm-symbolizer"
)

// llvmSymbolizer is a connection to an llvm-symbolizer command for
// obtaining address and line number information from a binary.
type llvmSymbolizer struct {
	sync.Mutex
	filename string
	rw       lineReaderWriter
	base     uint64
}

type llvmSymbolizerJob struct {
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
	// llvm-symbolizer requires the symbol type, CODE or DATA, for symbolization.
	symType string
}

func (a *llvmSymbolizerJob) write(s string) error {
	_, err := fmt.Fprintln(a.in, a.symType, s)
	return err
}

func (a *llvmSymbolizerJob) readLine() (string, error) {
	s, err := a.out.ReadString('\n')
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(s), nil
}

// close releases any resources used by the llvmSymbolizer object.
func (a *llvmSymbolizerJob) close() {
	a.in.Close()
	a.cmd.Wait()
}

// newLlvmSymbolizer starts the given llvmSymbolizer command reporting
// information about the given executable file. If file is a shared
// library, base should be the address at which it was mapped in the
// program under consideration.
func newLLVMSymbolizer(cmd, file string, base uint64, isData bool) (*llvmSymbolizer, error) {
	if cmd == "" {
		cmd = defaultLLVMSymbolizer
	}

	j := &llvmSymbolizerJob{
		cmd:     exec.Command(cmd, "--inlining", "-demangle=false"),
		symType: "CODE",
	}
	if isData {
		j.symType = "DATA"
	}

	var err error
	if j.in, err = j.cmd.StdinPipe(); err != nil {
		return nil, err
	}

	outPipe, err := j.cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	j.out = bufio.NewReader(outPipe)
	if err := j.cmd.Start(); err != nil {
		return nil, err
	}

	a := &llvmSymbolizer{
		filename: file,
		rw:       j,
		base:     base,
	}

	return a, nil
}

// readFrame parses the llvm-symbolizer output for a single address. It
// returns a populated plugin.Frame and whether it has reached the end of the
// data.
func (d *llvmSymbolizer) readFrame() (plugin.Frame, bool) {
	funcname, err := d.rw.readLine()
	if err != nil {
		return plugin.Frame{}, true
	}

	switch funcname {
	case "":
		return plugin.Frame{}, true
	case "??":
		funcname = ""
	}

	fileline, err := d.rw.readLine()
	if err != nil {
		return plugin.Frame{Func: funcname}, true
	}

	linenumber := 0
	// The llvm-symbolizer outputs the <file_name>:<line_number>:<column_number>.
	// When it cannot identify the source code location, it outputs "??:0:0".
	// Older versions output just the filename and line number, so we check for
	// both conditions here.
	if fileline == "??:0" || fileline == "??:0:0" {
		fileline = ""
	} else {
		switch split := strings.Split(fileline, ":"); len(split) {
		case 1:
			// filename
			fileline = split[0]
		case 2, 3:
			// filename:line , or
			// filename:line:disc , or
			fileline = split[0]
			if line, err := strconv.Atoi(split[1]); err == nil {
				linenumber = line
			}
		default:
			// Unrecognized, ignore
		}
	}

	return plugin.Frame{Func: funcname, File: fileline, Line: linenumber}, false
}

// addrInfo returns the stack frame information for a specific program
// address. It returns nil if the address could not be identified.
func (d *llvmSymbolizer) addrInfo(addr uint64) ([]plugin.Frame, error) {
	d.Lock()
	defer d.Unlock()

	if err := d.rw.write(fmt.Sprintf("%s 0x%x", d.filename, addr-d.base)); err != nil {
		return nil, err
	}

	var stack []plugin.Frame
	for {
		frame, end := d.readFrame()
		if end {
			break
		}

		if frame != (plugin.Frame{}) {
			stack = append(stack, frame)
		}
	}

	return stack, nil
}
