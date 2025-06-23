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
	defaultAddr2line = "addr2line"

	// addr2line may produce multiple lines of output. We
	// use this sentinel to identify the end of the output.
	sentinel = ^uint64(0)
)

// addr2Liner is a connection to an addr2line command for obtaining
// address and line number information from a binary.
type addr2Liner struct {
	mu   sync.Mutex
	rw   lineReaderWriter
	base uint64

	// nm holds an addr2Liner using nm tool. Certain versions of addr2line
	// produce incomplete names due to
	// https://sourceware.org/bugzilla/show_bug.cgi?id=17541. As a workaround,
	// the names from nm are used when they look more complete. See addrInfo()
	// code below for the exact heuristic.
	nm *addr2LinerNM
}

// lineReaderWriter is an interface to abstract the I/O to an addr2line
// process. It writes a line of input to the job, and reads its output
// one line at a time.
type lineReaderWriter interface {
	write(string) error
	readLine() (string, error)
	close()
}

type addr2LinerJob struct {
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
}

func (a *addr2LinerJob) write(s string) error {
	_, err := fmt.Fprint(a.in, s+"\n")
	return err
}

func (a *addr2LinerJob) readLine() (string, error) {
	s, err := a.out.ReadString('\n')
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(s), nil
}

// close releases any resources used by the addr2liner object.
func (a *addr2LinerJob) close() {
	a.in.Close()
	a.cmd.Wait()
}

// newAddr2Liner starts the given addr2liner command reporting
// information about the given executable file. If file is a shared
// library, base should be the address at which it was mapped in the
// program under consideration.
func newAddr2Liner(cmd, file string, base uint64) (*addr2Liner, error) {
	if cmd == "" {
		cmd = defaultAddr2line
	}

	j := &addr2LinerJob{
		cmd: exec.Command(cmd, "-aif", "-e", file),
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

	a := &addr2Liner{
		rw:   j,
		base: base,
	}

	return a, nil
}

// readFrame parses the addr2line output for a single address. It
// returns a populated plugin.Frame and whether it has reached the end of the
// data.
func (d *addr2Liner) readFrame() (plugin.Frame, bool) {
	funcname, err := d.rw.readLine()
	if err != nil {
		return plugin.Frame{}, true
	}
	if strings.HasPrefix(funcname, "0x") {
		// If addr2line returns a hex address we can assume it is the
		// sentinel. Read and ignore next two lines of output from
		// addr2line
		d.rw.readLine()
		d.rw.readLine()
		return plugin.Frame{}, true
	}

	fileline, err := d.rw.readLine()
	if err != nil {
		return plugin.Frame{}, true
	}

	linenumber := 0

	if funcname == "??" {
		funcname = ""
	}

	if fileline == "??:0" {
		fileline = ""
	} else {
		if i := strings.LastIndex(fileline, ":"); i >= 0 {
			// Remove discriminator, if present
			if disc := strings.Index(fileline, " (discriminator"); disc > 0 {
				fileline = fileline[:disc]
			}
			// If we cannot parse a number after the last ":", keep it as
			// part of the filename.
			if line, err := strconv.Atoi(fileline[i+1:]); err == nil {
				linenumber = line
				fileline = fileline[:i]
			}
		}
	}

	return plugin.Frame{
		Func: funcname,
		File: fileline,
		Line: linenumber}, false
}

func (d *addr2Liner) rawAddrInfo(addr uint64) ([]plugin.Frame, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if err := d.rw.write(fmt.Sprintf("%x", addr-d.base)); err != nil {
		return nil, err
	}

	if err := d.rw.write(fmt.Sprintf("%x", sentinel)); err != nil {
		return nil, err
	}

	resp, err := d.rw.readLine()
	if err != nil {
		return nil, err
	}

	if !strings.HasPrefix(resp, "0x") {
		return nil, fmt.Errorf("unexpected addr2line output: %s", resp)
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
	return stack, err
}

// addrInfo returns the stack frame information for a specific program
// address. It returns nil if the address could not be identified.
func (d *addr2Liner) addrInfo(addr uint64) ([]plugin.Frame, error) {
	stack, err := d.rawAddrInfo(addr)
	if err != nil {
		return nil, err
	}

	// Certain versions of addr2line produce incomplete names due to
	// https://sourceware.org/bugzilla/show_bug.cgi?id=17541. Attempt to replace
	// the name with a better one from nm.
	if len(stack) > 0 && d.nm != nil {
		nm, err := d.nm.addrInfo(addr)
		if err == nil && len(nm) > 0 {
			// Last entry in frame list should match since it is non-inlined. As a
			// simple heuristic, we only switch to the nm-based name if it is longer
			// by 2 or more characters. We consider nm names that are longer by 1
			// character insignificant to avoid replacing foo with _foo on MacOS (for
			// unknown reasons read2line produces the former and nm produces the
			// latter on MacOS even though both tools are asked to produce mangled
			// names).
			nmName := nm[len(nm)-1].Func
			a2lName := stack[len(stack)-1].Func
			if len(nmName) > len(a2lName)+1 {
				stack[len(stack)-1].Func = nmName
			}
		}
	}

	return stack, nil
}
