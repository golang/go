// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godebugs_test

import (
	"internal/godebugs"
	"internal/testenv"
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestAll(t *testing.T) {
	data, err := os.ReadFile("../../../doc/godebug.md")
	if err != nil {
		if os.IsNotExist(err) && (testenv.Builder() == "" || runtime.GOOS != "linux") {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	doc := string(data)

	last := ""
	for _, info := range godebugs.All {
		if info.Name <= last {
			t.Errorf("All not sorted: %s then %s", last, info.Name)
		}
		last = info.Name

		if info.Package == "" {
			t.Errorf("Name=%s missing Package", info.Name)
		}
		if info.Changed != 0 && info.Old == "" {
			t.Errorf("Name=%s has Changed, missing Old", info.Name)
		}
		if info.Old != "" && info.Changed == 0 {
			t.Errorf("Name=%s has Old, missing Changed", info.Name)
		}
		if !strings.Contains(doc, "`"+info.Name+"`") {
			t.Errorf("Name=%s not documented in doc/godebug.md", info.Name)
		}
	}
}
