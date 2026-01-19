// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetry manages the telemetry mode file.
package telemetry

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// Default is the default directory containing Go telemetry configuration and
// data.
//
// If Default is uninitialized, Default.Mode will be "off". As a consequence,
// no data should be written to the directory, and so the path values of
// LocalDir, UploadDir, etc. must not matter.
//
// Default is a global for convenience and testing, but should not be mutated
// outside of tests.
//
// TODO(rfindley): it would be nice to completely eliminate this global state,
// or at least push it in the golang.org/x/telemetry package
var Default Dir

// A Dir holds paths to telemetry data inside a directory.
type Dir struct {
	dir, local, upload, debug, modefile string
}

// NewDir creates a new Dir encapsulating paths in the given dir.
//
// NewDir does not create any new directories or files--it merely encapsulates
// the telemetry directory layout.
func NewDir(dir string) Dir {
	return Dir{
		dir:      dir,
		local:    filepath.Join(dir, "local"),
		upload:   filepath.Join(dir, "upload"),
		debug:    filepath.Join(dir, "debug"),
		modefile: filepath.Join(dir, "mode"),
	}
}

func init() {
	cfgDir, err := os.UserConfigDir()
	if err != nil {
		return
	}
	Default = NewDir(filepath.Join(cfgDir, "go", "telemetry"))
}

func (d Dir) Dir() string {
	return d.dir
}

func (d Dir) LocalDir() string {
	return d.local
}

func (d Dir) UploadDir() string {
	return d.upload
}

func (d Dir) DebugDir() string {
	return d.debug
}

func (d Dir) ModeFile() string {
	return d.modefile
}

// SetMode updates the telemetry mode with the given mode.
// Acceptable values for mode are "on", "off", or "local".
//
// SetMode always writes the mode file, and explicitly records the date at
// which the modefile was updated. This means that calling SetMode with "on"
// effectively resets the timeout before the next telemetry report is uploaded.
func (d Dir) SetMode(mode string) error {
	return d.SetModeAsOf(mode, time.Now())
}

// SetModeAsOf is like SetMode, but accepts an explicit time to use to
// back-date the mode state. This exists only for testing purposes.
func (d Dir) SetModeAsOf(mode string, asofTime time.Time) error {
	mode = strings.TrimSpace(mode)
	switch mode {
	case "on", "off", "local":
	default:
		return fmt.Errorf("invalid telemetry mode: %q", mode)
	}
	if d.modefile == "" {
		return fmt.Errorf("cannot determine telemetry mode file name")
	}
	// TODO(rfindley): why is this not 777, consistent with the use of 666 below?
	if err := os.MkdirAll(filepath.Dir(d.modefile), 0755); err != nil {
		return fmt.Errorf("cannot create a telemetry mode file: %w", err)
	}

	asof := asofTime.UTC().Format(DateOnly)
	// Defensively guarantee that we can parse the asof time.
	if _, err := time.Parse(DateOnly, asof); err != nil {
		return fmt.Errorf("internal error: invalid mode date %q: %v", asof, err)
	}

	data := []byte(mode + " " + asof)
	return os.WriteFile(d.modefile, data, 0666)
}

// Mode returns the current telemetry mode, as well as the time that the mode
// was effective.
//
// If there is no effective time, the second result is the zero time.
//
// If Mode is "off", no data should be written to the telemetry directory, and
// the other paths values referenced by Dir should be considered undefined.
// This accounts for the case where initializing [Default] fails, and therefore
// local telemetry paths are unknown.
func (d Dir) Mode() (string, time.Time) {
	if d.modefile == "" {
		return "off", time.Time{} // it's likely LocalDir/UploadDir are empty too. Turn off telemetry.
	}
	data, err := os.ReadFile(d.modefile)
	if err != nil {
		return "local", time.Time{} // default
	}
	mode := string(data)
	mode = strings.TrimSpace(mode)

	// Forward compatibility for https://go.dev/issue/63142#issuecomment-1734025130
	//
	// If the modefile contains a date, return it.
	if idx := strings.Index(mode, " "); idx >= 0 {
		d, err := time.Parse(DateOnly, mode[idx+1:])
		if err != nil {
			d = time.Time{}
		}
		return mode[:idx], d
	}

	return mode, time.Time{}
}

// DisabledOnPlatform indicates whether telemetry is disabled
// due to bugs in the current platform.
//
// TODO(rfindley): move to a more appropriate file.
const DisabledOnPlatform = false ||
	// The following platforms could potentially be supported in the future:
	runtime.GOOS == "openbsd" || // #60614
	runtime.GOOS == "solaris" || // #60968 #60970
	runtime.GOOS == "android" || // #60967
	runtime.GOOS == "illumos" || // #65544
	// These platforms fundamentally can't be supported:
	runtime.GOOS == "js" || // #60971
	runtime.GOOS == "wasip1" || // #60971
	runtime.GOOS == "plan9" || // https://github.com/golang/go/issues/57540#issuecomment-1470766639
	runtime.GOARCH == "mips" || runtime.GOARCH == "mipsle" // mips lacks cross-process 64-bit atomics
