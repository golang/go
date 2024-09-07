// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// coverMerge manages the state for merging test coverage profiles.
// It ensures thread-safe operations on a single coverage profile file
// across multiple test runs and packages.
var coverMerge struct {
	f          *os.File
	fsize      int64 // Tracks the size of valid data written to f
	sync.Mutex       // for f.Write
}

// initCoverProfile initializes the test coverage profile.
// It must be run before any calls to mergeCoverProfile or closeCoverProfile.
// Using this function clears the profile in case it existed from a previous run,
// or in case it doesn't exist and the test is going to fail to create it (or not run).
func initCoverProfile() {
	if testCoverProfile == "" || testC {
		return
	}
	if !filepath.IsAbs(testCoverProfile) {
		testCoverProfile = filepath.Join(testOutputDir.getAbs(), testCoverProfile)
	}

	// No mutex - caller's responsibility to call with no racing goroutines.
	f, err := os.Create(testCoverProfile)
	if err != nil {
		base.Fatalf("%v", err)
	}
	s, err := fmt.Fprintf(f, "mode: %s\n", cfg.BuildCoverMode)
	if err != nil {
		base.Fatalf("%v", err)
	}
	coverMerge.f = f
	coverMerge.fsize = int64(s)
}

// mergeCoverProfile merges file into the profile stored in testCoverProfile.
// It prints any errors it encounters to ew.
func mergeCoverProfile(file string) {
	if coverMerge.f == nil {
		return
	}
	coverMerge.Lock()
	defer coverMerge.Unlock()

	expect := fmt.Sprintf("mode: %s\n", cfg.BuildCoverMode)
	buf := make([]byte, len(expect))
	r, err := os.Open(file)
	if err != nil {
		// Test did not create profile, which is OK.
		return
	}
	defer r.Close()

	n, err := io.ReadFull(r, buf)
	if n == 0 {
		return
	}
	if err != nil || string(buf) != expect {
		base.Errorf("error: test wrote malformed coverage profile %s: header %q, expected %q: %v", file, string(buf), expect, err)
		return
	}
	s, err := io.Copy(coverMerge.f, r)
	if err != nil {
		base.Errorf("error: saving coverage profile: %v", err)
		return
	}
	coverMerge.fsize += s
}

func closeCoverProfile() {
	if coverMerge.f == nil {
		return
	}
	// Discard any partially written data from a failed merge.
	if err := coverMerge.f.Truncate(coverMerge.fsize); err != nil {
		base.Errorf("closing coverage profile: %v", err)
	}
	if err := coverMerge.f.Close(); err != nil {
		base.Errorf("closing coverage profile: %v", err)
	}
}
