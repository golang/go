// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bytes"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

// Test that TextHandler doesn't pull in the full encoding/json package.
func TestTextHandlerDependency(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tempDir := t.TempDir()
	main := filepath.Join(tempDir, "main.go")
	if err := os.WriteFile(main, []byte(`
package main
import (
	"log/slog"
	"os"
)
func main() {
	h := slog.NewTextHandler(os.Stderr, nil)
	l := slog.New(h)
	l.Info("hello")
}
`), 0644); err != nil {
		t.Fatal(err)
	}

	binary := filepath.Join(tempDir, "main.exe")
	goBin := testenv.GoToolPath(t)
	out, err := testenv.Command(t, goBin, "build", "-o", binary, main).CombinedOutput()
	if err != nil {
		t.Fatalf("go build failed: %v\n%s", err, out)
	}

	out, err = testenv.Command(t, goBin, "tool", "nm", binary).CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm failed: %v\n%s", err, out)
	}

	// We expect some encoding/json symbols due to initialization of global variables
	// in encoding/json, but we don't expect the JSON encoder or slog's JSON value
	// formatting to be linked in if we only use TextHandler.
	forbiddenSyms := []string{
		"log/slog.appendJSONMarshal",
		"encoding/json.(*Encoder).Encode",
	}

	for _, sym := range forbiddenSyms {
		if bytes.Contains(out, []byte(sym)) {
			t.Errorf("TextHandler program unexpectedly links in %q", sym)
		}
	}
}
