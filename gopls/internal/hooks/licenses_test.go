// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hooks

import (
	"crypto/sha256"
	"encoding/hex"
	"io/ioutil"
	"testing"
)

func TestLicenses(t *testing.T) {
	sumBytes, err := ioutil.ReadFile("../../go.sum")
	if err != nil {
		t.Fatal(err)
	}
	sumSum := sha256.Sum256(sumBytes)
	if licensesGeneratedFrom != hex.EncodeToString(sumSum[:]) {
		t.Error("combined license text needs updating. Run: `go generate ./internal/hooks` from the gopls module.")
	}
}
