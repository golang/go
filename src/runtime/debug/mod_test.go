// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug_test

import (
	"reflect"
	"runtime/debug"
	"strings"
	"testing"
)

// strip removes two leading tabs after each newline of s.
func strip(s string) string {
	replaced := strings.ReplaceAll(s, "\n\t\t", "\n")
	if len(replaced) > 0 && replaced[0] == '\n' {
		replaced = replaced[1:]
	}
	return replaced
}

func FuzzParseBuildInfoRoundTrip(f *testing.F) {
	// Package built from outside a module, missing some fields..
	f.Add(strip(`
		path	rsc.io/fortune
		mod	rsc.io/fortune	v1.0.0
		`))

	// Package built from the standard library, missing some fields..
	f.Add(`path	cmd/test2json`)

	// Package built from inside a module.
	f.Add(strip(`
		go	1.18
		path	example.com/m
		mod	example.com/m	(devel)	
		build	-compiler=gc
		`))

	// Package built in GOPATH mode.
	f.Add(strip(`
		go	1.18
		path	example.com/m
		build	-compiler=gc
		`))

	// Escaped build info.
	f.Add(strip(`
		go 1.18
		path example.com/m
		build CRAZY_ENV="requires\nescaping"
		`))

	f.Fuzz(func { t, s ->
		bi, err := debug.ParseBuildInfo(s)
		if err != nil {
			// Not a round-trippable BuildInfo string.
			t.Log(err)
			return
		}

		// s2 could have different escaping from s.
		// However, it should parse to exactly the same contents.
		s2 := bi.String()
		bi2, err := debug.ParseBuildInfo(s2)
		if err != nil {
			t.Fatalf("%v:\n%s", err, s2)
		}

		if !reflect.DeepEqual(bi2, bi) {
			t.Fatalf("Parsed representation differs.\ninput:\n%s\noutput:\n%s", s, s2)
		}
	})
}
