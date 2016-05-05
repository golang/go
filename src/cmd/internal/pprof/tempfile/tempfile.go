// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tempfile provides tools to create and delete temporary files
package tempfile

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// New returns an unused filename for output files.
func New(dir, prefix, suffix string) (*os.File, error) {
	for index := 1; index < 10000; index++ {
		path := filepath.Join(dir, fmt.Sprintf("%s%03d%s", prefix, index, suffix))
		if _, err := os.Stat(path); err != nil {
			return os.Create(path)
		}
	}
	// Give up
	return nil, fmt.Errorf("could not create file of the form %s%03d%s", prefix, 1, suffix)
}

var tempFiles []string
var tempFilesMu = sync.Mutex{}

// DeferDelete marks a file to be deleted by next call to Cleanup()
func DeferDelete(path string) {
	tempFilesMu.Lock()
	tempFiles = append(tempFiles, path)
	tempFilesMu.Unlock()
}

// Cleanup removes any temporary files selected for deferred cleaning.
func Cleanup() {
	tempFilesMu.Lock()
	for _, f := range tempFiles {
		os.Remove(f)
	}
	tempFiles = nil
	tempFilesMu.Unlock()
}
