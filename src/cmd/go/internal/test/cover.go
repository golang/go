// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"cmd/go/internal/base"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

var coverMerge struct {
	f          *os.File
	sync.Mutex // for f.Write
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
	_, err = fmt.Fprintf(f, "mode: %s\n", testCoverMode)
	if err != nil {
		base.Fatalf("%v", err)
	}
	coverMerge.f = f
}

// mergeCoverProfile merges file into the profile stored in testCoverProfile.
// It prints any errors it encounters to ew.
func mergeCoverProfile(ew io.Writer, file string) {
	if coverMerge.f == nil {
		return
	}
	coverMerge.Lock()
	defer coverMerge.Unlock()

	expect := fmt.Sprintf("mode: %s\n", testCoverMode)
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
		fmt.Fprintf(ew, "error: test wrote malformed coverage profile.\n")
		return
	}
	_, err = io.Copy(coverMerge.f, r)
	if err != nil {
		fmt.Fprintf(ew, "error: saving coverage profile: %v\n", err)
	}
}

func closeCoverProfile() {
	if coverMerge.f == nil {
		return
	}
	if err := coverMerge.f.Close(); err != nil {
		base.Errorf("closing coverage profile: %v", err)
	}
}
