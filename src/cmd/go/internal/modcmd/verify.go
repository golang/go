// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"

	"golang.org/x/mod/module"
	"golang.org/x/mod/sumdb/dirhash"
)

var cmdVerify = &base.Command{
	UsageLine: "go mod verify",
	Short:     "verify dependencies have expected content",
	Long: `
Verify checks that the dependencies of the current module,
which are stored in a local downloaded source cache, have not been
modified since being downloaded. If all the modules are unmodified,
verify prints "all modules verified." Otherwise it reports which
modules have been changed and causes 'go mod' to exit with a
non-zero status.
	`,
	Run: runVerify,
}

func init() {
	work.AddModCommonFlags(cmdVerify)
}

func runVerify(cmd *base.Command, args []string) {
	if len(args) != 0 {
		// NOTE(rsc): Could take a module pattern.
		base.Fatalf("go mod verify: verify takes no arguments")
	}
	// Checks go mod expected behavior
	if !modload.Enabled() || !modload.HasModRoot() {
		if cfg.Getenv("GO111MODULE") == "off" {
			base.Fatalf("go: modules disabled by GO111MODULE=off; see 'go help modules'")
		} else {
			base.Fatalf("go: cannot find main module; see 'go help modules'")
		}
	}
	ok := true
	for _, mod := range modload.LoadBuildList()[1:] {
		ok = verifyMod(mod) && ok
	}
	if ok {
		fmt.Printf("all modules verified\n")
	}
}

func verifyMod(mod module.Version) bool {
	ok := true
	zip, zipErr := modfetch.CachePath(mod, "zip")
	if zipErr == nil {
		_, zipErr = os.Stat(zip)
	}
	dir, dirErr := modfetch.DownloadDir(mod)
	data, err := ioutil.ReadFile(zip + "hash")
	if err != nil {
		if zipErr != nil && errors.Is(zipErr, os.ErrNotExist) &&
			dirErr != nil && errors.Is(dirErr, os.ErrNotExist) {
			// Nothing downloaded yet. Nothing to verify.
			return true
		}
		base.Errorf("%s %s: missing ziphash: %v", mod.Path, mod.Version, err)
		return false
	}
	h := string(bytes.TrimSpace(data))

	if zipErr != nil && errors.Is(zipErr, os.ErrNotExist) {
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
	if dirErr != nil && errors.Is(dirErr, os.ErrNotExist) {
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
