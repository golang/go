// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"runtime"

	"cmd/go/internal/base"
	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"

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

See https://golang.org/ref/mod#go-mod-verify for more about 'go mod verify'.
	`,
	Run: runVerify,
}

func init() {
	base.AddChdirFlag(&cmdVerify.Flag)
	base.AddModCommonFlags(&cmdVerify.Flag)
}

func runVerify(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	if len(args) != 0 {
		// NOTE(rsc): Could take a module pattern.
		base.Fatalf("go: verify takes no arguments")
	}
	modload.LoaderState.ForceUseModules = true
	modload.LoaderState.RootMode = modload.NeedRoot

	// Only verify up to GOMAXPROCS zips at once.
	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))

	mg, err := modload.LoadModGraph(ctx, "")
	if err != nil {
		base.Fatal(err)
	}
	mods := mg.BuildList()
	// Use a slice of result channels, so that the output is deterministic.
	errsChans := make([]<-chan []error, len(mods))

	for i, mod := range mods {
		sem <- token{}
		errsc := make(chan []error, 1)
		errsChans[i] = errsc
		mod := mod // use a copy to avoid data races
		go func() {
			errsc <- verifyMod(ctx, mod)
			<-sem
		}()
	}

	ok := true
	for _, errsc := range errsChans {
		errs := <-errsc
		for _, err := range errs {
			base.Errorf("%s", err)
			ok = false
		}
	}
	if ok {
		fmt.Printf("all modules verified\n")
	}
}

func verifyMod(ctx context.Context, mod module.Version) []error {
	if gover.IsToolchain(mod.Path) {
		// "go" and "toolchain" have no disk footprint; nothing to verify.
		return nil
	}
	if modload.LoaderState.MainModules.Contains(mod.Path) {
		return nil
	}
	var errs []error
	zip, zipErr := modfetch.CachePath(ctx, mod, "zip")
	if zipErr == nil {
		_, zipErr = os.Stat(zip)
	}
	dir, dirErr := modfetch.DownloadDir(ctx, mod)
	data, err := os.ReadFile(zip + "hash")
	if err != nil {
		if zipErr != nil && errors.Is(zipErr, fs.ErrNotExist) &&
			dirErr != nil && errors.Is(dirErr, fs.ErrNotExist) {
			// Nothing downloaded yet. Nothing to verify.
			return nil
		}
		errs = append(errs, fmt.Errorf("%s %s: missing ziphash: %v", mod.Path, mod.Version, err))
		return errs
	}
	h := string(bytes.TrimSpace(data))

	if zipErr != nil && errors.Is(zipErr, fs.ErrNotExist) {
		// ok
	} else {
		hZ, err := dirhash.HashZip(zip, dirhash.DefaultHash)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s %s: %v", mod.Path, mod.Version, err))
			return errs
		} else if hZ != h {
			errs = append(errs, fmt.Errorf("%s %s: zip has been modified (%v)", mod.Path, mod.Version, zip))
		}
	}
	if dirErr != nil && errors.Is(dirErr, fs.ErrNotExist) {
		// ok
	} else {
		hD, err := dirhash.HashDir(dir, mod.Path+"@"+mod.Version, dirhash.DefaultHash)
		if err != nil {

			errs = append(errs, fmt.Errorf("%s %s: %v", mod.Path, mod.Version, err))
			return errs
		}
		if hD != h {
			errs = append(errs, fmt.Errorf("%s %s: dir has been modified (%v)", mod.Path, mod.Version, dir))
		}
	}
	return errs
}
