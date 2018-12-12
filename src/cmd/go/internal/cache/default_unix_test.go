// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!darwin,!plan9

package cache

import (
	"os"
	"strings"
	"testing"
)

func TestDefaultDir(t *testing.T) {
	goCacheDir := "/tmp/test-go-cache"
	xdgCacheDir := "/tmp/test-xdg-cache"
	homeDir := "/tmp/test-home"

	// undo env changes when finished
	defer func(GOCACHE, XDG_CACHE_HOME, HOME string) {
		os.Setenv("GOCACHE", GOCACHE)
		os.Setenv("XDG_CACHE_HOME", XDG_CACHE_HOME)
		os.Setenv("HOME", HOME)
	}(os.Getenv("GOCACHE"), os.Getenv("XDG_CACHE_HOME"), os.Getenv("HOME"))

	os.Setenv("GOCACHE", goCacheDir)
	os.Setenv("XDG_CACHE_HOME", xdgCacheDir)
	os.Setenv("HOME", homeDir)

	dir, showWarnings := defaultDir()
	if dir != goCacheDir {
		t.Errorf("Cache DefaultDir %q should be $GOCACHE %q", dir, goCacheDir)
	}
	if !showWarnings {
		t.Error("Warnings should be shown when $GOCACHE is set")
	}

	os.Unsetenv("GOCACHE")
	dir, showWarnings = defaultDir()
	if !strings.HasPrefix(dir, xdgCacheDir+"/") {
		t.Errorf("Cache DefaultDir %q should be under $XDG_CACHE_HOME %q when $GOCACHE is unset", dir, xdgCacheDir)
	}
	if !showWarnings {
		t.Error("Warnings should be shown when $XDG_CACHE_HOME is set")
	}

	os.Unsetenv("XDG_CACHE_HOME")
	dir, showWarnings = defaultDir()
	if !strings.HasPrefix(dir, homeDir+"/.cache/") {
		t.Errorf("Cache DefaultDir %q should be under $HOME/.cache %q when $GOCACHE and $XDG_CACHE_HOME are unset", dir, homeDir+"/.cache")
	}
	if !showWarnings {
		t.Error("Warnings should be shown when $HOME is not /")
	}

	os.Unsetenv("HOME")
	if dir, _ := defaultDir(); dir != "off" {
		t.Error("Cache not disabled when $GOCACHE, $XDG_CACHE_HOME, and $HOME are unset")
	}

	os.Setenv("HOME", "/")
	if _, showWarnings := defaultDir(); showWarnings {
		// https://golang.org/issue/26280
		t.Error("Cache initialization warnings should be squelched when $GOCACHE and $XDG_CACHE_HOME are unset and $HOME is /")
	}
}
