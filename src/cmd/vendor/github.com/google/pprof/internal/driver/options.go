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

package driver

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/symbolizer"
	"github.com/google/pprof/internal/transport"
)

// setDefaults returns a new plugin.Options with zero fields sets to
// sensible defaults.
func setDefaults(o *plugin.Options) *plugin.Options {
	d := &plugin.Options{}
	if o != nil {
		*d = *o
	}
	if d.Writer == nil {
		d.Writer = oswriter{}
	}
	if d.Flagset == nil {
		d.Flagset = &GoFlags{}
	}
	if d.Obj == nil {
		d.Obj = &binutils.Binutils{}
	}
	if d.UI == nil {
		d.UI = &stdUI{r: bufio.NewReader(os.Stdin)}
	}
	if d.HTTPTransport == nil {
		d.HTTPTransport = transport.New(d.Flagset)
	}
	if d.Sym == nil {
		d.Sym = &symbolizer.Symbolizer{Obj: d.Obj, UI: d.UI, Transport: d.HTTPTransport}
	}
	return d
}

type stdUI struct {
	r *bufio.Reader
}

func (ui *stdUI) ReadLine(prompt string) (string, error) {
	os.Stdout.WriteString(prompt)
	return ui.r.ReadString('\n')
}

func (ui *stdUI) Print(args ...interface{}) {
	ui.fprint(os.Stderr, args)
}

func (ui *stdUI) PrintErr(args ...interface{}) {
	ui.fprint(os.Stderr, args)
}

func (ui *stdUI) IsTerminal() bool {
	return false
}

func (ui *stdUI) WantBrowser() bool {
	return true
}

func (ui *stdUI) SetAutoComplete(func(string) string) {
}

func (ui *stdUI) fprint(f *os.File, args []interface{}) {
	text := fmt.Sprint(args...)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	f.WriteString(text)
}

// oswriter implements the Writer interface using a regular file.
type oswriter struct{}

func (oswriter) Open(name string) (io.WriteCloser, error) {
	f, err := os.Create(name)
	return f, err
}
