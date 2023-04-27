// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package copyright

import (
	"strings"
	"testing"
)

func TestToolsCopyright(t *testing.T) {
	files, err := checkCopyright("..")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) > 0 {
		t.Errorf("The following files are missing copyright notices:\n%s", strings.Join(files, "\n"))
	}
}
