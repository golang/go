// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetry manages the telemetry mode file.
package telemetry

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// The followings are the process' default Settings.
// The values are subdirectories and a file under
// os.UserConfigDir()/go/telemetry.
// For convenience, each field is made to global
// and they are not supposed to be changed.
var (
	// Default directory containing count files and local reports (not yet uploaded)
	LocalDir string
	// Default directory containing uploaded reports.
	UploadDir string
	// Default file path that holds the telemetry mode info.
	ModeFile ModeFilePath
)

// ModeFilePath is the telemetry mode file path with methods to manipulate the file contents.
type ModeFilePath string

func init() {
	cfgDir, err := os.UserConfigDir()
	if err != nil {
		return
	}
	gotelemetrydir := filepath.Join(cfgDir, "go", "telemetry")
	LocalDir = filepath.Join(gotelemetrydir, "local")
	UploadDir = filepath.Join(gotelemetrydir, "upload")
	ModeFile = ModeFilePath(filepath.Join(gotelemetrydir, "mode"))
}

// SetMode updates the telemetry mode with the given mode.
// Acceptable values for mode are "on", "off", or "local".
//
// SetMode always writes the mode file, and explicitly records the date at
// which the modefile was updated. This means that calling SetMode with "on"
// effectively resets the timeout before the next telemetry report is uploaded.
func SetMode(mode string) error {
	return ModeFile.SetMode(mode)
}

func (m ModeFilePath) SetMode(mode string) error {
	return m.SetModeAsOf(mode, time.Now())
}

// SetModeAsOf is like SetMode, but accepts an explicit time to use to
// back-date the mode state. This exists only for testing purposes.
func (m ModeFilePath) SetModeAsOf(mode string, asofTime time.Time) error {
	mode = strings.TrimSpace(mode)
	switch mode {
	case "on", "off", "local":
	default:
		return fmt.Errorf("invalid telemetry mode: %q", mode)
	}
	fname := string(m)
	if fname == "" {
		return fmt.Errorf("cannot determine telemetry mode file name")
	}
	if err := os.MkdirAll(filepath.Dir(fname), 0755); err != nil {
		return fmt.Errorf("cannot create a telemetry mode file: %w", err)
	}

	asof := asofTime.UTC().Format("2006-01-02")
	// Defensively guarantee that we can parse the asof time.
	if _, err := time.Parse("2006-01-02", asof); err != nil {
		return fmt.Errorf("internal error: invalid mode date %q: %v", asof, err)
	}

	data := []byte(mode + " " + asof)
	return os.WriteFile(fname, data, 0666)
}

// Mode returns the current telemetry mode, as well as the time that the mode
// was effective.
//
// If there is no effective time, the second result is the zero time.
func Mode() (string, time.Time) {
	return ModeFile.Mode()
}

func (m ModeFilePath) Mode() (string, time.Time) {
	fname := string(m)
	if fname == "" {
		return "off", time.Time{} // it's likely LocalDir/UploadDir are empty too. Turn off telemetry.
	}
	data, err := os.ReadFile(fname)
	if err != nil {
		return "local", time.Time{} // default
	}
	mode := string(data)
	mode = strings.TrimSpace(mode)

	// Forward compatibility for https://go.dev/issue/63142#issuecomment-1734025130
	//
	// If the modefile contains a date, return it.
	if idx := strings.Index(mode, " "); idx >= 0 {
		d, err := time.Parse("2006-01-02", mode[idx+1:])
		if err != nil {
			d = time.Time{}
		}
		return mode[:idx], d
	}

	return mode, time.Time{}
}
