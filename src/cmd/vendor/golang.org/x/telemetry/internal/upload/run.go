// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upload

import (
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"golang.org/x/telemetry/internal/telemetry"
)

var logger *log.Logger

func init() {
	logger = log.New(io.Discard, "", 0)
}

// keep track of what SetLogOutput has seen
var seenlogwriters []io.Writer

// SetLogOutput sets the default logger's output destination.
func SetLogOutput(logging io.Writer) {
	if logging == nil {
		return
	}
	logger.SetOutput(logging) // the common case
	seenlogwriters = append(seenlogwriters, logging)
	if len(seenlogwriters) > 1 {
		// The client asked for logging, and there is also a debug dir
		logger.SetOutput(io.MultiWriter(seenlogwriters...))
	}
}

// LogIfDebug arranges to write a log file in the directory
// dirname, if it exists. If dirname is the empty string,
// the function tries the directory it.Localdir/debug.
func LogIfDebug(dirname string) error {
	dname := filepath.Join(telemetry.LocalDir, "debug")
	if dirname != "" {
		dname = dirname
	}
	fd, err := os.Stat(dname)
	if err != nil {
		return err
	}
	if fd == nil || !fd.IsDir() {
		// debug doesn't exist or isn't a directory
		return nil
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return fmt.Errorf("no build info")
	}
	year, month, day := time.Now().UTC().Date()
	goVers := info.GoVersion
	// E.g.,  goVers:"go1.22-20240109-RC01 cl/597041403 +dcbe772469 X:loopvar"
	words := strings.Fields(goVers)
	goVers = words[0]
	progPkgPath := info.Path
	if progPkgPath == "" {
		progPkgPath = strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	}
	prog := path.Base(progPkgPath)
	progVers := info.Main.Version
	fname := filepath.Join(dname, fmt.Sprintf("%s-%s-%s-%4d%02d%02d-%d.log",
		prog, progVers, goVers, year, month, day, os.Getpid()))
	fname = strings.ReplaceAll(fname, " ", "")
	if _, err := os.Stat(fname); err == nil {
		// This process previously called upload.Run
		return nil
	}
	logfd, err := os.Create(fname)
	if err != nil {
		return err
	}
	SetLogOutput(logfd)
	return nil
}

// Uploader carries parameters needed for upload.
type Uploader struct {
	// Config is used to select counters to upload.
	Config *telemetry.UploadConfig
	// ConfigVersion is the version of the config.
	ConfigVersion string

	// LocalDir is where the local counter files are.
	LocalDir string
	// UploadDir is where uploader leaves the copy of uploaded data.
	UploadDir string
	// ModeFilePath is the file.
	ModeFilePath telemetry.ModeFilePath

	UploadServerURL string
	StartTime       time.Time

	cache parsedCache
}

// NewUploader creates a default uploader.
func NewUploader(config *telemetry.UploadConfig) *Uploader {
	return &Uploader{
		Config:          config,
		ConfigVersion:   "custom",
		LocalDir:        telemetry.LocalDir,
		UploadDir:       telemetry.UploadDir,
		ModeFilePath:    telemetry.ModeFile,
		UploadServerURL: "https://telemetry.go.dev/upload",
		StartTime:       time.Now().UTC(),
	}
}

// Run generates and uploads reports
func (u *Uploader) Run() {
	if telemetry.DisabledOnPlatform {
		return
	}
	todo := u.findWork()
	ready, err := u.reports(&todo)
	if err != nil {
		logger.Printf("reports: %v", err)
	}
	for _, f := range ready {
		u.uploadReport(f)
	}
}
