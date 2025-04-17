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
	"encoding/json"
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
	isData   bool
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

// newLLVMSymbolizer starts the given llvmSymbolizer command reporting
// information about the given executable file. If file is a shared
// library, base should be the address at which it was mapped in the
// program under consideration.
func newLLVMSymbolizer(cmd, file string, base uint64, isData bool) (*llvmSymbolizer, error) {
	if cmd == "" {
		cmd = defaultLLVMSymbolizer
	}

	j := &llvmSymbolizerJob{
		cmd:     exec.Command(cmd, "--inlining", "-demangle=false", "--output-style=JSON"),
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
		isData:   isData,
	}

	return a, nil
}

// readDataFrames parses the llvm-symbolizer DATA output for a single address. It
// returns a populated plugin.Frame array with a single entry.
func (d *llvmSymbolizer) readDataFrames() ([]plugin.Frame, error) {
	line, err := d.rw.readLine()
	if err != nil {
		return nil, err
	}
	var frame struct {
		Address    string `json:"Address"`
		ModuleName string `json:"ModuleName"`
		Data       struct {
			Start string `json:"Start"`
			Size  string `json:"Size"`
			Name  string `json:"Name"`
		} `json:"Data"`
	}
	if err := json.Unmarshal([]byte(line), &frame); err != nil {
		return nil, err
	}
	// Match non-JSON output behaviour of stuffing the start/size into the filename of a single frame,
	// with the size being a decimal value.
	size, err := strconv.ParseInt(frame.Data.Size, 0, 0)
	if err != nil {
		return nil, err
	}
	var stack []plugin.Frame
	stack = append(stack, plugin.Frame{Func: frame.Data.Name, File: fmt.Sprintf("%s %d", frame.Data.Start, size)})
	return stack, nil
}

// readCodeFrames parses the llvm-symbolizer CODE output for a single address. It
// returns a populated plugin.Frame array.
func (d *llvmSymbolizer) readCodeFrames() ([]plugin.Frame, error) {
	line, err := d.rw.readLine()
	if err != nil {
		return nil, err
	}
	var frame struct {
		Address    string `json:"Address"`
		ModuleName string `json:"ModuleName"`
		Symbol     []struct {
			Line          int    `json:"Line"`
			Column        int    `json:"Column"`
			FunctionName  string `json:"FunctionName"`
			FileName      string `json:"FileName"`
			StartLine     int    `json:"StartLine"`
		} `json:"Symbol"`
	}
	if err := json.Unmarshal([]byte(line), &frame); err != nil {
		return nil, err
	}
	var stack []plugin.Frame
	for _, s := range frame.Symbol {
		stack = append(stack, plugin.Frame{Func: s.FunctionName, File: s.FileName, Line: s.Line, Column: s.Column, StartLine: s.StartLine})
	}
	return stack, nil
}

// addrInfo returns the stack frame information for a specific program
// address. It returns nil if the address could not be identified.
func (d *llvmSymbolizer) addrInfo(addr uint64) ([]plugin.Frame, error) {
	d.Lock()
	defer d.Unlock()

	if err := d.rw.write(fmt.Sprintf("%s 0x%x", d.filename, addr-d.base)); err != nil {
		return nil, err
	}
	if d.isData {
		return d.readDataFrames()
	}
	return d.readCodeFrames()
}
