// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package diff implements a Diff function that compare two inputs
// using the 'diff' tool.
package diff

import (
	"bytes"
	exec "internal/execabs"
	"io/ioutil"
	"os"
	"runtime"
)

// Returns diff of two arrays of bytes in diff tool format.
func Diff(prefix string, b1, b2 []byte) ([]byte, error) {
	f1, err := writeTempFile(prefix, b1)
	if err != nil {
		return nil, err
	}
	defer os.Remove(f1)

	f2, err := writeTempFile(prefix, b2)
	if err != nil {
		return nil, err
	}
	defer os.Remove(f2)

	cmd := "diff"
	if runtime.GOOS == "plan9" {
		cmd = "/bin/ape/diff"
	}

	data, err := exec.Command(cmd, "-u", f1, f2).CombinedOutput()
	if len(data) > 0 {
		// diff exits with a non-zero status when the files don't match.
		// Ignore that failure as long as we get output.
		err = nil
	}

	// If we are on Windows and the diff is Cygwin diff,
	// machines can get into a state where every Cygwin
	// command works fine but prints a useless message like:
	//
	//	Cygwin WARNING:
	//	  Couldn't compute FAST_CWD pointer.  This typically occurs if you're using
	//	  an older Cygwin version on a newer Windows.  Please update to the latest
	//	  available Cygwin version from https://cygwin.com/.  If the problem persists,
	//	  please see https://cygwin.com/problems.html
	//
	// Skip over that message and just return the actual diff.
	if len(data) > 0 && !bytes.HasPrefix(data, []byte("--- ")) {
		i := bytes.Index(data, []byte("\n--- "))
		if i >= 0 && i < 80*10 && bytes.Contains(data[:i], []byte("://cygwin.com/")) {
			data = data[i+1:]
		}
	}

	return data, err
}

func writeTempFile(prefix string, data []byte) (string, error) {
	file, err := ioutil.TempFile("", prefix)
	if err != nil {
		return "", err
	}
	_, err = file.Write(data)
	if err1 := file.Close(); err == nil {
		err = err1
	}
	if err != nil {
		os.Remove(file.Name())
		return "", err
	}
	return file.Name(), nil
}
