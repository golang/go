// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"net/url"
	"os"
	"strings"

	"golang.org/x/tools/internal/lsp/browser"
	"golang.org/x/tools/internal/lsp/debug"
)

// version implements the version command.
type version struct {
	app *Application
}

func (v *version) Name() string      { return "version" }
func (v *version) Usage() string     { return "" }
func (v *version) ShortHelp() string { return "print the gopls version information" }
func (v *version) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), ``)
	f.PrintDefaults()
}

// Run prints version information to stdout.
func (v *version) Run(ctx context.Context, args ...string) error {
	debug.PrintVersionInfo(os.Stdout, v.app.Verbose, debug.PlainText)
	return nil
}

// bug implements the bug command.
type bug struct{}

func (b *bug) Name() string      { return "bug" }
func (b *bug) Usage() string     { return "" }
func (b *bug) ShortHelp() string { return "report a bug in gopls" }
func (b *bug) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), ``)
	f.PrintDefaults()
}

const goplsBugPrefix = "x/tools/gopls: "
const goplsBugHeader = `Please answer these questions before submitting your issue. Thanks!

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
	debug.PrintVersionInfo(buf, true, debug.Markdown)
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
