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

	"golang.org/x/telemetry/internal/configstore"
	"golang.org/x/telemetry/internal/telemetry"
)

// RunConfig configures non-default behavior of a call to Run.
//
// All fields are optional, for testing or observability.
type RunConfig struct {
	TelemetryDir string    // if set, overrides the telemetry data directory
	UploadURL    string    // if set, overrides the telemetry upload endpoint
	LogWriter    io.Writer // if set, used for detailed logging of the upload process
	Env          []string  // if set, appended to the config download environment
	StartTime    time.Time // if set, overrides the upload start time
}

// Uploader encapsulates a single upload operation, carrying parameters and
// shared state.
type Uploader struct {
	// config is used to select counters to upload.
	config        *telemetry.UploadConfig //
	configVersion string                  // version of the config
	dir           telemetry.Dir           // the telemetry dir to process

	uploadServerURL string
	startTime       time.Time

	cache parsedCache

	logFile *os.File
	logger  *log.Logger
}

// NewUploader creates a new uploader to use for running the upload for the
// given config.
//
// Uploaders should only be used for one call to [Run].
func NewUploader(rcfg RunConfig) (*Uploader, error) {
	// Determine the upload directory.
	var dir telemetry.Dir
	if rcfg.TelemetryDir != "" {
		dir = telemetry.NewDir(rcfg.TelemetryDir)
	} else {
		dir = telemetry.Default
	}

	// Determine the upload URL.
	uploadURL := rcfg.UploadURL
	if uploadURL == "" {
		uploadURL = "https://telemetry.go.dev/upload"
	}

	// Determine the upload logger.
	//
	// This depends on the provided rcfg.LogWriter and the presence of
	// dir.DebugDir, as follows:
	//  1. If LogWriter is present, log to it.
	//  2. If DebugDir is present, log to a file within it.
	//  3. If both LogWriter and DebugDir are present, log to a multi writer.
	//  4. If neither LogWriter nor DebugDir are present, log to a noop logger.
	var logWriters []io.Writer
	logFile, err := debugLogFile(dir.DebugDir())
	if err != nil {
		logFile = nil
	}
	if logFile != nil {
		logWriters = append(logWriters, logFile)
	}
	if rcfg.LogWriter != nil {
		logWriters = append(logWriters, rcfg.LogWriter)
	}
	var logWriter io.Writer
	switch len(logWriters) {
	case 0:
		logWriter = io.Discard
	case 1:
		logWriter = logWriters[0]
	default:
		logWriter = io.MultiWriter(logWriters...)
	}
	logger := log.New(logWriter, "", log.Ltime|log.Lmicroseconds|log.Lshortfile)

	// Fetch the upload config, if it is not provided.
	config, configVersion, err := configstore.Download("latest", rcfg.Env)
	if err != nil {
		return nil, err
	}

	// Set the start time, if it is not provided.
	startTime := time.Now().UTC()
	if !rcfg.StartTime.IsZero() {
		startTime = rcfg.StartTime
	}

	return &Uploader{
		config:          config,
		configVersion:   configVersion,
		dir:             dir,
		uploadServerURL: uploadURL,
		startTime:       startTime,

		logFile: logFile,
		logger:  logger,
	}, nil
}

// Close cleans up any resources associated with the uploader.
func (u *Uploader) Close() error {
	if u.logFile == nil {
		return nil
	}
	return u.logFile.Close()
}

// Run generates and uploads reports
func (u *Uploader) Run() error {
	if telemetry.DisabledOnPlatform {
		return nil
	}
	todo := u.findWork()
	ready, err := u.reports(&todo)
	if err != nil {
		u.logger.Printf("Error building reports: %v", err)
		return fmt.Errorf("reports failed: %v", err)
	}
	u.logger.Printf("Uploading %d reports", len(ready))
	for _, f := range ready {
		u.uploadReport(f)
	}
	return nil
}

// debugLogFile arranges to write a log file in the given debug directory, if
// it exists.
func debugLogFile(debugDir string) (*os.File, error) {
	fd, err := os.Stat(debugDir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if !fd.IsDir() {
		return nil, fmt.Errorf("debug path %q is not a directory", debugDir)
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return nil, fmt.Errorf("no build info")
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
	fname := filepath.Join(debugDir, fmt.Sprintf("%s-%s-%s-%4d%02d%02d-%d.log",
		prog, progVers, goVers, year, month, day, os.Getpid()))
	fname = strings.ReplaceAll(fname, " ", "")
	if _, err := os.Stat(fname); err == nil {
		// This process previously called upload.Run
		return nil, nil
	}
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0666)
	if err != nil {
		if os.IsExist(err) {
			return nil, nil // this process previously called upload.Run
		}
		return nil, err
	}
	return f, nil
}
