// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteDiskCache(t *testing.T) {
	ctx := context.Background()

	tmpdir, err := os.MkdirTemp("", "go-writeCache-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	err = writeDiskCache(ctx, filepath.Join(tmpdir, "file"), []byte("data"))
	if err != nil {
		t.Fatal(err)
	}
}
