// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"unicode"
)

var (
	nl         = []byte("\n")
	slashSlash = []byte("//")
	plusBuild  = []byte("+build")
)

// checkBuildTag checks that build tags are in the correct location and well-formed.
func checkBuildTag(name string, data []byte) {
	if !vet("buildtags") {
		return
	}
	lines := bytes.SplitAfter(data, nl)

	// Determine cutpoint where +build comments are no longer valid.
	// They are valid in leading // comments in the file followed by
	// a blank line.
	var cutoff int
	for i, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			cutoff = i
			continue
		}
		if bytes.HasPrefix(line, slashSlash) {
			continue
		}
		break
	}

	for i, line := range lines {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, slashSlash) {
			continue
		}
		text := bytes.TrimSpace(line[2:])
		if bytes.HasPrefix(text, plusBuild) {
			fields := bytes.Fields(text)
			if !bytes.Equal(fields[0], plusBuild) {
				// Comment is something like +buildasdf not +build.
				fmt.Fprintf(os.Stderr, "%s:%d: possible malformed +build comment\n", name, i+1)
				continue
			}
			if i >= cutoff {
				fmt.Fprintf(os.Stderr, "%s:%d: +build comment must appear before package clause and be followed by a blank line\n", name, i+1)
				setExit(1)
				continue
			}
			// Check arguments.
		Args:
			for _, arg := range fields[1:] {
				for _, elem := range strings.Split(string(arg), ",") {
					if strings.HasPrefix(elem, "!!") {
						fmt.Fprintf(os.Stderr, "%s:%d: invalid double negative in build constraint: %s\n", name, i+1, arg)
						setExit(1)
						break Args
					}
					if strings.HasPrefix(elem, "!") {
						elem = elem[1:]
					}
					for _, c := range elem {
						if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
							fmt.Fprintf(os.Stderr, "%s:%d: invalid non-alphanumeric build constraint: %s\n", name, i+1, arg)
							setExit(1)
							break Args
						}
					}
				}
			}
			continue
		}
		// Comment with +build but not at beginning.
		if bytes.Contains(line, plusBuild) && i < cutoff {
			fmt.Fprintf(os.Stderr, "%s:%d: possible malformed +build comment\n", name, i+1)
			continue
		}
	}
}
