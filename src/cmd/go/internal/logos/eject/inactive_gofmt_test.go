// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
)

func TestInactiveVariantsFormatEachOutputOnce(t *testing.T) {
	ejectTestSaveToolEnv(t)
	saved := ejectGofmtFiles
	t.Cleanup(func() { ejectGofmtFiles = saved })
	ejectToolExeSuffix = ""
	root := t.TempDir()
	bin := filepath.Join(root, "bin")
	if err := os.Mkdir(bin, 0777); err != nil {
		t.Fatal(err)
	}
	log := filepath.Join(root, "formatted.txt")
	t.Setenv("EJECT_TEST_GOFMT_LOG", log)
	ejectTestWriteScript(t, filepath.Join(bin, "gofmt"), "#!/bin/sh\nif [ \"$1\" = -w ]; then\nshift\nprintf '%s\\n' \"$@\" >> \"$EJECT_TEST_GOFMT_LOG\"\nfi\n")
	cfg.GOROOT, cfg.GOBIN = root, t.TempDir()
	// One source may appear in several actual variant load inventories. Its
	// output must never be passed twice to parallel gofmt -w workers.
	ejectGofmtFiles = []string{"b.go", "a.go", "b.go", "a.go"}
	runEjectGofmt(t.Context())
	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Fields(string(data))
	if !slices.Equal(got, []string{"a.go", "b.go"}) {
		t.Fatalf("formatter received duplicate or unstable variant paths: %q", got)
	}
}
