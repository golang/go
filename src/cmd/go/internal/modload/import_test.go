// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"internal/testenv"
	"regexp"
	"strings"
	"testing"
)

var importTests = []struct {
	path string
	err  string
}{
	{
		path: "golang.org/x/net/context",
		err:  "missing module for import: golang.org/x/net@.* provides golang.org/x/net/context",
	},
	{
		path: "golang.org/x/net",
		err:  "cannot find module providing package golang.org/x/net",
	},
	{
		path: "golang.org/x/text",
		err:  "missing module for import: golang.org/x/text@.* provides golang.org/x/text",
	},
	{
		path: "github.com/rsc/quote/buggy",
		err:  "missing module for import: github.com/rsc/quote@v1.5.2 provides github.com/rsc/quote/buggy",
	},
	{
		path: "github.com/rsc/quote",
		err:  "missing module for import: github.com/rsc/quote@v1.5.2 provides github.com/rsc/quote",
	},
	{
		path: "golang.org/x/foo/bar",
		err:  "cannot find module providing package golang.org/x/foo/bar",
	},
}

func TestImport(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	for _, tt := range importTests {
		t.Run(strings.Replace(tt.path, "/", "_", -1), func(t *testing.T) {
			// Note that there is no build list, so Import should always fail.
			m, dir, err := Import(tt.path)
			if err == nil {
				t.Fatalf("Import(%q) = %v, %v, nil; expected error", tt.path, m, dir)
			}
			if !regexp.MustCompile(tt.err).MatchString(err.Error()) {
				t.Fatalf("Import(%q): error %q, want error matching %#q", tt.path, err, tt.err)
			}
		})
	}
}
