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

// pprof is a tool for collection, manipulation and visualization
// of performance profiles.
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/chzyer/readline"
	"github.com/google/pprof/driver"
)

func main() {
	if err := driver.PProf(&driver.Options{UI: newUI()}); err != nil {
		fmt.Fprintf(os.Stderr, "pprof: %v\n", err)
		os.Exit(2)
	}
}

// readlineUI implements the driver.UI interface using the
// github.com/chzyer/readline library.
// This is contained in pprof.go to avoid adding the readline
// dependency in the vendored copy of pprof in the Go distribution,
// which does not use this file.
type readlineUI struct {
	rl *readline.Instance
}

func newUI() driver.UI {
	rl, err := readline.New("")
	if err != nil {
		fmt.Fprintf(os.Stderr, "readline: %v", err)
		return nil
	}
	return &readlineUI{
		rl: rl,
	}
}

// Read returns a line of text (a command) read from the user.
// prompt is printed before reading the command.
func (r *readlineUI) ReadLine(prompt string) (string, error) {
	r.rl.SetPrompt(prompt)
	return r.rl.Readline()
}

// Print shows a message to the user.
// It is printed over stderr as stdout is reserved for regular output.
func (r *readlineUI) Print(args ...interface{}) {
	text := fmt.Sprint(args...)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	fmt.Fprint(r.rl.Stderr(), text)
}

// Print shows a message to the user, colored in red for emphasis.
// It is printed over stderr as stdout is reserved for regular output.
func (r *readlineUI) PrintErr(args ...interface{}) {
	text := fmt.Sprint(args...)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	fmt.Fprint(r.rl.Stderr(), colorize(text))
}

// colorize the msg using ANSI color escapes.
func colorize(msg string) string {
	var red = 31
	var colorEscape = fmt.Sprintf("\033[0;%dm", red)
	var colorResetEscape = "\033[0m"
	return colorEscape + msg + colorResetEscape
}

// IsTerminal returns whether the UI is known to be tied to an
// interactive terminal (as opposed to being redirected to a file).
func (r *readlineUI) IsTerminal() bool {
	const stdout = 1
	return readline.IsTerminal(stdout)
}

// Start a browser on interactive mode.
func (r *readlineUI) WantBrowser() bool {
	return r.IsTerminal()
}

// SetAutoComplete instructs the UI to call complete(cmd) to obtain
// the auto-completion of cmd, if the UI supports auto-completion at all.
func (r *readlineUI) SetAutoComplete(complete func(string) string) {
	// TODO: Implement auto-completion support.
}
