// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package debug exports debug information for gopls.
package debug

import (
	"context"
	"fmt"
	"io"
	"strings"

	"golang.org/x/tools/internal/gocommand"
)

type PrintMode int

const (
	PlainText = PrintMode(iota)
	Markdown
	HTML
)

// Version is a manually-updated mechanism for tracking versions.
var Version = "master"

// PrintServerInfo writes HTML debug info to w for the Instance.
func (i *Instance) PrintServerInfo(ctx context.Context, w io.Writer) {
	section(w, HTML, "Server Instance", func() {
		fmt.Fprintf(w, "Start time: %v\n", i.StartTime)
		fmt.Fprintf(w, "LogFile: %s\n", i.Logfile)
		fmt.Fprintf(w, "Working directory: %s\n", i.Workdir)
		fmt.Fprintf(w, "Address: %s\n", i.ServerAddress)
		fmt.Fprintf(w, "Debug address: %s\n", i.DebugAddress)
	})
	PrintVersionInfo(ctx, w, true, HTML)
}

// PrintVersionInfo writes version information to w, using the output format
// specified by mode. verbose controls whether additional information is
// written, including section headers.
func PrintVersionInfo(ctx context.Context, w io.Writer, verbose bool, mode PrintMode) {
	if !verbose {
		printBuildInfo(w, false, mode)
		return
	}
	section(w, mode, "Build info", func() {
		printBuildInfo(w, true, mode)
	})
	fmt.Fprint(w, "\n")
	section(w, mode, "Go info", func() {
		gocmdRunner := &gocommand.Runner{}
		version, err := gocmdRunner.Run(ctx, gocommand.Invocation{
			Verb: "version",
		})
		if err != nil {
			panic(err)
		}
		fmt.Fprintln(w, version.String())
	})
}

func section(w io.Writer, mode PrintMode, title string, body func()) {
	switch mode {
	case PlainText:
		fmt.Fprintln(w, title)
		fmt.Fprintln(w, strings.Repeat("-", len(title)))
		body()
	case Markdown:
		fmt.Fprintf(w, "#### %s\n\n```\n", title)
		body()
		fmt.Fprintf(w, "```\n")
	case HTML:
		fmt.Fprintf(w, "<h3>%s</h3>\n<pre>\n", title)
		body()
		fmt.Fprint(w, "</pre>\n")
	}
}
