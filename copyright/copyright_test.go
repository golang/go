// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package copyright

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestToolsCopyright(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tools := filepath.Dir(cwd)
	if !strings.HasSuffix(filepath.Base(tools), "tools") {
		t.Fatalf("current working directory is %s, expected tools", tools)
	}
	files, err := checkCopyright(tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) > 0 {
		t.Errorf("The following files are missing copyright notices:\n%s", strings.Join(files, "\n"))
	}
}
