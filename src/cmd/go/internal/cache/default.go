// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"cmd/go/internal/base"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

// Default returns the default cache to use, or nil if no cache should be used.
func Default() *Cache {
	defaultOnce.Do(initDefaultCache)
	return defaultCache
}

var (
	defaultOnce  sync.Once
	defaultCache *Cache
)

// initDefaultCache does the work of finding the default cache
// the first time Default is called.
func initDefaultCache() {
	dir := os.Getenv("GOCACHE")
	if dir == "off" {
		return
	}
	if dir == "" {
		// Compute default location.
		// TODO(rsc): This code belongs somewhere else,
		// like maybe ioutil.CacheDir or os.CacheDir.
		switch runtime.GOOS {
		case "windows":
			dir = os.Getenv("LocalAppData")

		case "darwin":
			dir = os.Getenv("HOME")
			if dir == "" {
				return
			}
			dir += "/Library/Caches"

		case "plan9":
			dir = os.Getenv("home")
			if dir == "" {
				return
			}
			dir += "/lib/cache"

		default: // Unix
			// https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
			dir = os.Getenv("XDG_CACHE_HOME")
			if dir == "" {
				dir = os.Getenv("HOME")
				if dir == "" {
					return
				}
				dir += "/.cache"
			}
		}
		dir = filepath.Join(dir, "go-build")
		if err := os.MkdirAll(dir, 0777); err != nil {
			return
		}
	}

	c, err := Open(dir)
	if err != nil {
		base.Fatalf("initializing cache in $GOCACHE: %s", err)
	}
	defaultCache = c
}
