// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package renameio writes files atomically by renaming temporary files.
package renameio

import (
	"bytes"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

const patternSuffix = ".tmp"

// Pattern returns a glob pattern that matches the unrenamed temporary files
// created when writing to filename.
func Pattern(filename string) string {
	return filepath.Join(filepath.Dir(filename), filepath.Base(filename)+patternSuffix)
}

// WriteFile is like ioutil.WriteFile, but first writes data to an arbitrary
// file in the same directory as filename, then renames it atomically to the
// final name.
//
// That ensures that the final location, if it exists, is always a complete file.
func WriteFile(filename string, data []byte, perm os.FileMode) (err error) {
	return WriteToFile(filename, bytes.NewReader(data), perm)
}

// WriteToFile is a variant of WriteFile that accepts the data as an io.Reader
// instead of a slice.
func WriteToFile(filename string, data io.Reader, perm os.FileMode) (err error) {
	f, err := tempFile(filepath.Dir(filename), filepath.Base(filename), perm)
	if err != nil {
		return err
	}
	defer func() {
		// Only call os.Remove on f.Name() if we failed to rename it: otherwise,
		// some other process may have created a new file with the same name after
		// that.
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()

	if _, err := io.Copy(f, data); err != nil {
		return err
	}
	// Sync the file before renaming it: otherwise, after a crash the reader may
	// observe a 0-length file instead of the actual contents.
	// See https://golang.org/issue/22397#issuecomment-380831736.
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	var start time.Time
	for {
		err := os.Rename(f.Name(), filename)
		if err == nil || !isAccessDeniedError(err) {
			return err
		}

		// Windows seems to occasionally trigger spurious "Access is denied" errors
		// here (see golang.org/issue/31247). We're not sure why. It's probably
		// worth a little extra latency to avoid propagating the spurious errors.
		if start.IsZero() {
			start = time.Now()
		} else if time.Since(start) >= 500*time.Millisecond {
			return err
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// tempFile creates a new temporary file with given permission bits.
func tempFile(dir, prefix string, perm os.FileMode) (f *os.File, err error) {
	for i := 0; i < 10000; i++ {
		name := filepath.Join(dir, prefix+strconv.Itoa(rand.Intn(1000000000))+patternSuffix)
		f, err = os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, perm)
		if os.IsExist(err) {
			continue
		}
		break
	}
	return
}
