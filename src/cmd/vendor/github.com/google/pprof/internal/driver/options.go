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
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/symbolizer"
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
		d.Flagset = goFlags{}
	}
	if d.Obj == nil {
		d.Obj = &binutils.Binutils{}
	}
	if d.UI == nil {
		d.UI = &stdUI{r: bufio.NewReader(os.Stdin)}
	}
	if d.Sym == nil {
		d.Sym = &symbolizer.Symbolizer{Obj: d.Obj, UI: d.UI}
	}
	return d
}

// goFlags returns a flagset implementation based on the standard flag
// package from the Go distribution. It implements the plugin.FlagSet
// interface.
type goFlags struct{}

func (goFlags) Bool(o string, d bool, c string) *bool {
	return flag.Bool(o, d, c)
}

func (goFlags) Int(o string, d int, c string) *int {
	return flag.Int(o, d, c)
}

func (goFlags) Float64(o string, d float64, c string) *float64 {
	return flag.Float64(o, d, c)
}

func (goFlags) String(o, d, c string) *string {
	return flag.String(o, d, c)
}

func (goFlags) BoolVar(b *bool, o string, d bool, c string) {
	flag.BoolVar(b, o, d, c)
}

func (goFlags) IntVar(i *int, o string, d int, c string) {
	flag.IntVar(i, o, d, c)
}

func (goFlags) Float64Var(f *float64, o string, d float64, c string) {
	flag.Float64Var(f, o, d, c)
}

func (goFlags) StringVar(s *string, o, d, c string) {
	flag.StringVar(s, o, d, c)
}

func (goFlags) StringList(o, d, c string) *[]*string {
	return &[]*string{flag.String(o, d, c)}
}

func (goFlags) ExtraUsage() string {
	return ""
}

func (goFlags) Parse(usage func()) []string {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		usage()
	}
	return args
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
