// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vgo

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"cmd/go/internal/base"
	"cmd/go/internal/dirhash"
	"cmd/go/internal/module"
)

var CmdVerify = &base.Command{
	UsageLine: "verify",
	Run:       runVerify,
	Short:     "verify downloaded modules against expected hashes",
	Long: `
Verify checks that the dependencies of the current module,
which are stored in a local downloaded source cache,
have not been modified since being downloaded.

If all the modules are unmodified, verify prints

	all modules verified

and exits successfully (status 0). Otherwise, verify reports
which modules have been changed and exits with a non-zero status.
	`,
}

func runVerify(cmd *base.Command, args []string) {
	if Init(); !Enabled() {
		base.Fatalf("vgo verify: cannot use outside module")
	}
	if len(args) != 0 {
		// TODO: take arguments
		base.Fatalf("vgo verify: verify takes no arguments")
	}

	// Make go.mod consistent but don't load any packages.
	InitMod()
	iterate(func(*loader) {})
	writeGoMod()

	ok := true
	for _, mod := range buildList[1:] {
		ok = verifyMod(mod) && ok
	}
	if ok {
		fmt.Printf("all modules verified\n")
	}
}

func verifyMod(mod module.Version) bool {
	ok := true
	zip := filepath.Join(srcV, "cache", mod.Path, "/@v/", mod.Version+".zip")
	_, zipErr := os.Stat(zip)
	dir := filepath.Join(srcV, mod.Path+"@"+mod.Version)
	_, dirErr := os.Stat(dir)
	data, err := ioutil.ReadFile(zip + "hash")
	if err != nil {
		if zipErr != nil && os.IsNotExist(zipErr) && dirErr != nil && os.IsNotExist(dirErr) {
			// Nothing downloaded yet. Nothing to verify.
			return true
		}
		base.Errorf("%s %s: missing ziphash: %v", mod.Path, mod.Version, err)
		return false
	}
	h := string(bytes.TrimSpace(data))

	if zipErr != nil && os.IsNotExist(zipErr) {
		// ok
	} else {
		hZ, err := dirhash.HashZip(zip, dirhash.DefaultHash)
		if err != nil {
			base.Errorf("%s %s: %v", mod.Path, mod.Version, err)
			return false
		} else if hZ != h {
			base.Errorf("%s %s: zip has been modified (%v)", mod.Path, mod.Version, zip)
			ok = false
		}
	}
	if dirErr != nil && os.IsNotExist(dirErr) {
		// ok
	} else {
		hD, err := dirhash.HashDir(dir, mod.Path+"@"+mod.Version, dirhash.DefaultHash)
		if err != nil {

			base.Errorf("%s %s: %v", mod.Path, mod.Version, err)
			return false
		}
		if hD != h {
			base.Errorf("%s %s: dir has been modified (%v)", mod.Path, mod.Version, dir)
			ok = false
		}
	}
	return ok
}
