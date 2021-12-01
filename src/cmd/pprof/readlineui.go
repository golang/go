// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains a driver.UI implementation
// that provides the readline functionality if possible.

//go:build (darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows) && !appengine && !android

package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/google/pprof/driver"
	"golang.org/x/term"
)

func init() {
	newUI = newReadlineUI
}

// readlineUI implements driver.UI interface using the
// golang.org/x/term package.
// The upstream pprof command implements the same functionality
// using the github.com/chzyer/readline package.
type readlineUI struct {
	term *term.Terminal
}

func newReadlineUI() driver.UI {
	// disable readline UI in dumb terminal. (golang.org/issue/26254)
	if v := strings.ToLower(os.Getenv("TERM")); v == "" || v == "dumb" {
		return nil
	}
	// test if we can use term.ReadLine
	// that assumes operation in the raw mode.
	oldState, err := term.MakeRaw(0)
	if err != nil {
		return nil
	}
	term.Restore(0, oldState)

	rw := struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stderr}
	return &readlineUI{term: term.NewTerminal(rw, "")}
}

// Read returns a line of text (a command) read from the user.
// prompt is printed before reading the command.
func (r *readlineUI) ReadLine(prompt string) (string, error) {
	r.term.SetPrompt(prompt)

	// skip error checking because we tested it
	// when creating this readlineUI initially.
	oldState, _ := term.MakeRaw(0)
	defer term.Restore(0, oldState)

	s, err := r.term.ReadLine()
	return s, err
}

// Print shows a message to the user.
// It formats the text as fmt.Print would and adds a final \n if not already present.
// For line-based UI, Print writes to standard error.
// (Standard output is reserved for report data.)
func (r *readlineUI) Print(args ...any) {
	r.print(false, args...)
}

// PrintErr shows an error message to the user.
// It formats the text as fmt.Print would and adds a final \n if not already present.
// For line-based UI, PrintErr writes to standard error.
func (r *readlineUI) PrintErr(args ...any) {
	r.print(true, args...)
}

func (r *readlineUI) print(withColor bool, args ...any) {
	text := fmt.Sprint(args...)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	if withColor {
		text = colorize(text)
	}
	fmt.Fprint(r.term, text)
}

// colorize prints the msg in red using ANSI color escapes.
func colorize(msg string) string {
	const red = 31
	var colorEscape = fmt.Sprintf("\033[0;%dm", red)
	var colorResetEscape = "\033[0m"
	return colorEscape + msg + colorResetEscape
}

// IsTerminal reports whether the UI is known to be tied to an
// interactive terminal (as opposed to being redirected to a file).
func (r *readlineUI) IsTerminal() bool {
	const stdout = 1
	return term.IsTerminal(stdout)
}

// WantBrowser indicates whether browser should be opened with the -http option.
func (r *readlineUI) WantBrowser() bool {
	return r.IsTerminal()
}

// SetAutoComplete instructs the UI to call complete(cmd) to obtain
// the auto-completion of cmd, if the UI supports auto-completion at all.
func (r *readlineUI) SetAutoComplete(complete func(string) string) {
	// TODO: Implement auto-completion support.
}
