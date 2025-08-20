// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"internal/testenv"
	"regexp"
	"strings"
	"testing"

	"golang.org/x/mod/module"
)

var importTests = []struct {
	path string
	m    module.Version
	err  string
}{
	{
		path: "golang.org/x/net/context",
		m: module.Version{
			Path: "golang.org/x/net",
		},
	},
	{
		path: "golang.org/x/net",
		err:  `module golang.org/x/net@.* found \(v[01]\.\d+\.\d+\), but does not contain package golang.org/x/net`,
	},
	{
		path: "golang.org/x/text",
		m: module.Version{
			Path: "golang.org/x/text",
		},
	},
	{
		path: "github.com/rsc/quote/buggy",
		m: module.Version{
			Path:    "github.com/rsc/quote",
			Version: "v1.5.2",
		},
	},
	{
		path: "github.com/rsc/quote",
		m: module.Version{
			Path:    "github.com/rsc/quote",
			Version: "v1.5.2",
		},
	},
	{
		path: "golang.org/x/foo/bar",
		err:  "cannot find module providing package golang.org/x/foo/bar",
	},
}

func TestQueryImport(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	oldAllowMissingModuleImports := allowMissingModuleImports
	oldRootMode := LoaderState.RootMode
	defer func() {
		allowMissingModuleImports = oldAllowMissingModuleImports
		LoaderState.RootMode = oldRootMode
	}()
	allowMissingModuleImports = true
	LoaderState.RootMode = NoRoot

	ctx := context.Background()
	rs := LoadModFile(ctx)

	for _, tt := range importTests {
		t.Run(strings.ReplaceAll(tt.path, "/", "_"), func(t *testing.T) {
			// Note that there is no build list, so Import should always fail.
			m, err := queryImport(ctx, tt.path, rs)

			if tt.err == "" {
				if err != nil {
					t.Fatalf("queryImport(_, %q): %v", tt.path, err)
				}
			} else {
				if err == nil {
					t.Fatalf("queryImport(_, %q) = %v, nil; expected error", tt.path, m)
				}
				if !regexp.MustCompile(tt.err).MatchString(err.Error()) {
					t.Fatalf("queryImport(_, %q): error %q, want error matching %#q", tt.path, err, tt.err)
				}
			}

			if m.Path != tt.m.Path || (tt.m.Version != "" && m.Version != tt.m.Version) {
				t.Errorf("queryImport(_, %q) = %v, _; want %v", tt.path, m, tt.m)
			}
		})
	}
}
