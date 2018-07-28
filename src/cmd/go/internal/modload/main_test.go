// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"testing"

	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
)

func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	dir, err := ioutil.TempDir("", "modload-test-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	modfetch.SrcMod = filepath.Join(dir, "src/mod")
	codehost.WorkRoot = filepath.Join(dir, "codework")
	return m.Run()
}
