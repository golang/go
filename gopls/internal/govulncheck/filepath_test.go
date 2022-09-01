// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAbsRelShorter(t *testing.T) {
	thisFile := "filepath_test.go"
	thisFileAbs, _ := filepath.Abs(thisFile)

	tf, err := os.CreateTemp("", "filepath_test.gp")
	if err != nil {
		t.Errorf("could not create temporary filepath_test.go file: %v", err)
	}
	tempFile := tf.Name()
	tempFileAbs, _ := filepath.Abs(tempFile)

	for _, test := range []struct {
		l    string
		want string
	}{
		{thisFile, "filepath_test.go"},
		{thisFileAbs, "filepath_test.go"},
		// Relative path to temp file from "." is longer as
		// it needs to go back the length of the absolute
		// path and then in addition go to os.TempDir.
		{tempFile, tempFileAbs},
	} {
		if got := AbsRelShorter(test.l); got != test.want {
			t.Errorf("want %s; got %s", test.want, got)
		}
	}
}
