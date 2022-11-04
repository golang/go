// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rechecker

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRechecker(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "file")
	r := &Rechecker[string]{
		File:     tmpFile,
		Duration: time.Minute,
		Parse: func(content []byte) (*string, error) {
			str := string(content)
			return &str, nil
		},
	}

	fileContent := "content"
	if err := os.WriteFile(tmpFile, []byte(fileContent), 0660); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 5; i++ {
		val, err := r.Get()
		if err != nil {
			t.Fatalf("%v: %v", i, err)
		}

		if *val != fileContent {
			t.Fatalf("%v: expected %v, got %v", i, fileContent, *val)
		}
	}

	r.lastCheched = r.lastCheched.Add(-time.Minute)
	r.modTime = r.modTime.Add(-time.Minute)

	fileContent = "new file content"
	if err := os.WriteFile(tmpFile, []byte(fileContent), 0660); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 5; i++ {
		val, err := r.Get()
		if err != nil {
			t.Fatalf("content update: %v: %v", i, err)
		}

		if *val != fileContent {
			t.Fatalf("content update: %v: expected %v, got %v", i, fileContent, *val)
		}
	}
}
