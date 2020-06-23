// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteDiskCache(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "go-writeCache-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	err = writeDiskCache(filepath.Join(tmpdir, "file"), []byte("data"))
	if err != nil {
		t.Fatal(err)
	}
}
