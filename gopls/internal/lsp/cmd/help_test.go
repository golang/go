// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"bytes"
	"context"
	"flag"
	"io/ioutil"
	"path/filepath"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/cmd"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/internal/tool"
)

//go:generate go test -run Help -update-help-files

var updateHelpFiles = flag.Bool("update-help-files", false, "Write out the help files instead of checking them")

const appName = "gopls"

func TestHelpFiles(t *testing.T) {
	testenv.NeedsGoBuild(t) // This is a lie. We actually need the source code.
	app := cmd.New(appName, "", nil, nil)
	ctx := context.Background()
	for _, page := range append(app.Commands(), app) {
		t.Run(page.Name(), func(t *testing.T) {
			var buf bytes.Buffer
			s := flag.NewFlagSet(page.Name(), flag.ContinueOnError)
			s.SetOutput(&buf)
			tool.Run(ctx, s, page, []string{"-h"})
			name := page.Name()
			if name == appName {
				name = "usage"
			}
			helpFile := filepath.Join("usage", name+".hlp")
			got := buf.Bytes()
			if *updateHelpFiles {
				if err := ioutil.WriteFile(helpFile, got, 0666); err != nil {
					t.Errorf("Failed writing %v: %v", helpFile, err)
				}
				return
			}
			expect, err := ioutil.ReadFile(helpFile)
			switch {
			case err != nil:
				t.Errorf("Missing help file %q", helpFile)
			case !bytes.Equal(expect, got):
				t.Errorf("Help file %q did not match, got:\n%q\nwant:\n%q", helpFile, string(got), string(expect))
			}
		})
	}
}
