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

// Package proftest provides some utility routines to test other
// packages related to profiles.
package proftest

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"testing"
)

// Diff compares two byte arrays using the diff tool to highlight the
// differences. It is meant for testing purposes to display the
// differences between expected and actual output.
func Diff(b1, b2 []byte) (data []byte, err error) {
	f1, err := ioutil.TempFile("", "proto_test")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f1.Name())
	defer f1.Close()

	f2, err := ioutil.TempFile("", "proto_test")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f2.Name())
	defer f2.Close()

	f1.Write(b1)
	f2.Write(b2)

	data, err = exec.Command("diff", "-u", f1.Name(), f2.Name()).CombinedOutput()
	if len(data) > 0 {
		// diff exits with a non-zero status when the files don't match.
		// Ignore that failure as long as we get output.
		err = nil
	}
	if err != nil {
		data = []byte(fmt.Sprintf("diff failed: %v\nb1: %q\nb2: %q\n", err, b1, b2))
		err = nil
	}
	return
}

// EncodeJSON encodes a value into a byte array. This is intended for
// testing purposes.
func EncodeJSON(x interface{}) []byte {
	data, err := json.MarshalIndent(x, "", "    ")
	if err != nil {
		panic(err)
	}
	data = append(data, '\n')
	return data
}

// TestUI implements the plugin.UI interface, triggering test failures
// if more than Ignore errors are printed.
type TestUI struct {
	T      *testing.T
	Ignore int
}

// ReadLine returns no input, as no input is expected during testing.
func (ui *TestUI) ReadLine(_ string) (string, error) {
	return "", fmt.Errorf("no input")
}

// Print messages are discarded by the test UI.
func (ui *TestUI) Print(args ...interface{}) {
}

// PrintErr messages may trigger an error failure. A fixed number of
// error messages are permitted when appropriate.
func (ui *TestUI) PrintErr(args ...interface{}) {
	if ui.Ignore > 0 {
		ui.Ignore--
		return
	}
	ui.T.Error(args)
}

// IsTerminal indicates if the UI is an interactive terminal.
func (ui *TestUI) IsTerminal() bool {
	return false
}

// SetAutoComplete is not supported by the test UI.
func (ui *TestUI) SetAutoComplete(_ func(string) string) {
}
