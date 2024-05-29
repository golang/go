// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package genflags

import (
	"flag"
	"strings"
	"testing"
)

// ShortTestFlags returns the set of "-test." flag shorthand names that end
// users may pass to 'go test'.
func ShortTestFlags() []string {
	testing.Init()

	var names []string
	flag.VisitAll(func(f *flag.Flag) {
		var name string
		var found bool
		if name, found = strings.CutPrefix(f.Name, "test."); !found {
			return
		}

		switch name {
		case "testlogfile", "paniconexit0", "paniconexit", "fuzzcachedir", "fuzzworker", "gocoverdir":
			// These flags are only for use by cmd/go.
		default:
			names = append(names, name)
		}
	})

	return names
}
