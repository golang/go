// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/url"
	"os"
	"strings"

	"golang.org/x/tools/internal/lsp/browser"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/source"
)

// version implements the version command.
type version struct {
	JSON bool `flag:"json" help:"outputs in json format."`

	app *Application
}

func (v *version) Name() string      { return "version" }
func (v *version) Parent() string    { return v.app.Name() }
func (v *version) Usage() string     { return "" }
func (v *version) ShortHelp() string { return "print the gopls version information" }
func (v *version) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), ``)
	printFlagDefaults(f)
}

// Run prints version information to stdout.
func (v *version) Run(ctx context.Context, args ...string) error {
	var mode = debug.PlainText
	if v.JSON {
		mode = debug.JSON
	}

	return debug.PrintVersionInfo(ctx, os.Stdout, v.app.verbose(), mode)
}

// bug implements the bug command.
type bug struct {
	app *Application
}

func (b *bug) Name() string      { return "bug" }
func (b *bug) Parent() string    { return b.app.Name() }
func (b *bug) Usage() string     { return "" }
func (b *bug) ShortHelp() string { return "report a bug in gopls" }
func (b *bug) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), ``)
	printFlagDefaults(f)
}

const goplsBugPrefix = "x/tools/gopls: <DESCRIBE THE PROBLEM>"
const goplsBugHeader = `ATTENTION: Please answer these questions BEFORE submitting your issue. Thanks!

#### What did you do?
If possible, provide a recipe for reproducing the error.
A complete runnable program is good.
A link on play.golang.org is better.
A failing unit test is the best.

#### What did you expect to see?


#### What did you see instead?


`

// Run collects some basic information and then prepares an issue ready to
// be reported.
func (b *bug) Run(ctx context.Context, args ...string) error {
	buf := &bytes.Buffer{}
	fmt.Fprint(buf, goplsBugHeader)
	debug.PrintVersionInfo(ctx, buf, true, debug.Markdown)
	body := buf.String()
	title := strings.Join(args, " ")
	if !strings.HasPrefix(title, goplsBugPrefix) {
		title = goplsBugPrefix + title
	}
	if !browser.Open("https://github.com/golang/go/issues/new?title=" + url.QueryEscape(title) + "&body=" + url.QueryEscape(body)) {
		fmt.Print("Please file a new issue at golang.org/issue/new using this template:\n\n")
		fmt.Print(body)
	}
	return nil
}

type apiJSON struct {
	app *Application
}

func (j *apiJSON) Name() string      { return "api-json" }
func (j *apiJSON) Parent() string    { return j.app.Name() }
func (j *apiJSON) Usage() string     { return "" }
func (j *apiJSON) ShortHelp() string { return "print json describing gopls API" }
func (j *apiJSON) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), ``)
	printFlagDefaults(f)
}

func (j *apiJSON) Run(ctx context.Context, args ...string) error {
	js, err := json.MarshalIndent(source.GeneratedAPIJSON, "", "\t")
	if err != nil {
		return err
	}
	fmt.Fprint(os.Stdout, string(js))
	return nil
}

type licenses struct {
	app *Application
}

func (l *licenses) Name() string      { return "licenses" }
func (l *licenses) Parent() string    { return l.app.Name() }
func (l *licenses) Usage() string     { return "" }
func (l *licenses) ShortHelp() string { return "print licenses of included software" }
func (l *licenses) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), ``)
	printFlagDefaults(f)
}

const licensePreamble = `
gopls is made available under the following BSD-style license:

Copyright (c) 2009 The Go Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

gopls implements the LSP specification, which is made available under the following license:

Copyright (c) Microsoft Corporation

All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

gopls also includes software made available under these licenses:
`

func (l *licenses) Run(ctx context.Context, args ...string) error {
	opts := source.DefaultOptions()
	l.app.options(opts)
	txt := licensePreamble
	if opts.LicensesText == "" {
		txt += "(development gopls, license information not available)"
	} else {
		txt += opts.LicensesText
	}
	fmt.Fprint(os.Stdout, txt)
	return nil
}
