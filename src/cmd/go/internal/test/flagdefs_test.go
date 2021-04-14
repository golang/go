// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"flag"
	"strings"
	"testing"
)

func TestPassFlagToTestIncludesAllTestFlags(t *testing.T) {
	flag.VisitAll(func(f *flag.Flag) {
		if !strings.HasPrefix(f.Name, "test.") {
			return
		}
		name := strings.TrimPrefix(f.Name, "test.")
		switch name {
		case "testlogfile", "paniconexit0":
			// These are internal flags.
		default:
			if !passFlagToTest[name] {
				t.Errorf("passFlagToTest missing entry for %q (flag test.%s)", name, name)
				t.Logf("(Run 'go generate cmd/go/internal/test' if it should be added.)")
			}
		}
	})

	for name := range passFlagToTest {
		if flag.Lookup("test."+name) == nil {
			t.Errorf("passFlagToTest contains %q, but flag -test.%s does not exist in test binary", name, name)
		}

		if CmdTest.Flag.Lookup(name) == nil {
			t.Errorf("passFlagToTest contains %q, but flag -%s does not exist in 'go test' subcommand", name, name)
		}
	}
}
