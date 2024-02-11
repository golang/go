// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"internal/goexperiment"
)

// Default returns the default cache to use.
// It never returns nil.
func Default() Cache {
	defaultOnce.Do(initDefaultCache)
	return defaultCache
}

var (
	defaultOnce  sync.Once
	defaultCache Cache
)

// cacheREADME is a message stored in a README in the cache directory.
// Because the cache lives outside the normal Go trees, we leave the
// README as a courtesy to explain where it came from.
const cacheREADME = `This directory holds cached build artifacts from the Go build system.
Run "go clean -cache" if the directory is getting too large.
Run "go clean -fuzzcache" to delete the fuzz cache.
See golang.org to learn more about Go.
`

// initDefaultCache does the work of finding the default cache
// the first time Default is called.
func initDefaultCache() {
	dir, _ := DefaultDir()
	if dir == "off" {
		if defaultDirErr != nil {
			base.Fatalf("build cache is required, but could not be located: %v", defaultDirErr)
		}
		base.Fatalf("build cache is disabled by GOCACHE=off, but required as of Go 1.12")
	}
	if err := os.MkdirAll(dir, 0777); err != nil {
		base.Fatalf("failed to initialize build cache at %s: %s\n", dir, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "README")); err != nil {
		// Best effort.
		os.WriteFile(filepath.Join(dir, "README"), []byte(cacheREADME), 0666)
	}

	diskCache, err := Open(dir)
	if err != nil {
		base.Fatalf("failed to initialize build cache at %s: %s\n", dir, err)
	}

	if v := cfg.Getenv("GOCACHEPROG"); v != "" && goexperiment.CacheProg {
		defaultCache = startCacheProg(v, diskCache)
	} else {
		defaultCache = diskCache
	}
}

var (
	defaultDirOnce    sync.Once
	defaultDir        string
	defaultDirChanged bool
	defaultDirErr     error
)

// DefaultDir returns the effective GOCACHE setting.
// It returns "off" if the cache is disabled.
func DefaultDir() (string, bool) {
	// Save the result of the first call to DefaultDir for later use in
	// initDefaultCache. cmd/go/main.go explicitly sets GOCACHE so that
	// subprocesses will inherit it, but that means initDefaultCache can't
	// otherwise distinguish between an explicit "off" and a UserCacheDir error.

	defaultDirOnce.Do(func() {
		defaultDir = cfg.Getenv("GOCACHE")
		if filepath.IsAbs(defaultDir) || defaultDir == "off" {
			defaultDirChanged = true
			return
		}
		if defaultDir != "" {
			defaultDir = "off"
			defaultDirChanged = true
			defaultDirErr = fmt.Errorf("GOCACHE is not an absolute path")
			return
		}

		// Compute default location.
		dir, err := os.UserCacheDir()
		if err != nil {
			defaultDir = "off"
			defaultDirErr = fmt.Errorf("GOCACHE is not defined and %v", err)
			return
		}
		defaultDir = filepath.Join(dir, "go-build")
	})

	return defaultDir, defaultDirChanged
}
