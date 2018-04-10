// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfs_test

import (
	"os"
	"runtime"
	"testing"

	"golang.org/x/tools/godoc/vfs"
)

func TestRootType(t *testing.T) {
	goPath := os.Getenv("GOPATH")
	var expectedType vfs.RootType
	if goPath == "" {
		expectedType = ""
	} else {
		expectedType = vfs.RootTypeGoPath
	}
	tests := []struct {
		path   string
		fsType vfs.RootType
	}{
		{runtime.GOROOT(), vfs.RootTypeGoRoot},
		{goPath, expectedType},
		{"/tmp/", ""},
	}

	for _, item := range tests {
		fs := vfs.OS(item.path)
		if fs.RootType("path") != item.fsType {
			t.Errorf("unexpected fsType. Expected- %v, Got- %v", item.fsType, fs.RootType("path"))
		}
	}
}
