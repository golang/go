// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upload

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/telemetry/internal/config"
	"golang.org/x/telemetry/internal/configstore"
	"golang.org/x/telemetry/internal/counter"
	"golang.org/x/telemetry/internal/telemetry"
)

// reports generates reports from inactive count files
func (u *Uploader) reports(todo *work) ([]string, error) {
	if mode, _ := u.ModeFilePath.Mode(); mode == "off" {
		return nil, nil // no reports
	}
	thisInstant := u.StartTime
	today := thisInstant.Format("2006-01-02")
	lastWeek := latestReport(todo.uploaded)
	if lastWeek >= today { //should never happen
		lastWeek = ""
	}
	logger.Printf("lastWeek %q, today %s", lastWeek, today)
	countFiles := make(map[string][]string) // expiry date string->filenames
	earliest := make(map[string]time.Time)  // earliest begin time for any counter
	for _, f := range todo.countfiles {
		begin, end := u.counterDateSpan(f)

		if end.Before(thisInstant) {
			expiry := end.Format(dateFormat)
			countFiles[expiry] = append(countFiles[expiry], f)
			if earliest[expiry].IsZero() || earliest[expiry].After(begin) {
				earliest[expiry] = begin
			}
		}
	}
	for expiry, files := range countFiles {
		if notNeeded(expiry, *todo) {
			logger.Printf("files for %s not needed, deleting %v", expiry, files)
			// The report already exists.
			// There's another check in createReport.
			deleteFiles(files)
			continue
		}
		fname, err := u.createReport(earliest[expiry], expiry, files, lastWeek)
		if err != nil {
			return nil, err
		}
		if fname != "" {
			todo.readyfiles = append(todo.readyfiles, fname)
		}
	}
	return todo.readyfiles, nil
}

// latestReport returns the YYYY-MM-DD of the last report uploaded
// or the empty string if there are no reports.
func latestReport(uploaded map[string]bool) string {
	var latest string
	for name := range uploaded {
		if strings.HasSuffix(name, ".json") {
			if name > latest {
				latest = name
			}
		}
	}
	if latest == "" {
		return ""
	}
	// strip off the .json
	return latest[:len(latest)-len(".json")]
}

// notNeeded returns true if the report for date has already been created
func notNeeded(date string, todo work) bool {
	if todo.uploaded != nil && todo.uploaded[date+".json"] {
		return true
	}
	// maybe the report is already in todo.readyfiles
	for _, f := range todo.readyfiles {
		if strings.Contains(f, date) {
			return true
		}
	}
	return false
}

func deleteFiles(files []string) {
	for _, f := range files {
		if err := os.Remove(f); err != nil {
			// this could be a race condition.
			// conversely, on Windows, err may be nil and
			// the file not deleted if anyone has it open.
			logger.Printf("%v failed to remove %s", err, f)
		}
	}
}

// createReport for all the count files for the same date.
// returns the absolute path name of the file containing the report
func (u *Uploader) createReport(start time.Time, expiryDate string, files []string, lastWeek string) (string, error) {
	if u.Config == nil {
		a, v, err := configstore.Download("latest", nil)
		if err != nil {
			logger.Print(err) // or something (e.g., panic(err))
		}
		u.Config = &a
		u.ConfigVersion = v
	}
	uploadOK := true
	mode, asof := u.ModeFilePath.Mode()
	if u.Config == nil || mode != "on" {
		logger.Printf("no upload config or mode %q is not 'on'", mode)
		uploadOK = false // no config, nothing to upload
	}
	if tooOld(expiryDate, u.StartTime) {
		logger.Printf("expiryDate %s is too old", expiryDate)
		uploadOK = false
	}
	// If the mode is recorded with an asof date, don't upload if the report
	// includes any data on or before the asof date.
	if !asof.IsZero() && !asof.Before(start) {
		logger.Printf("asof %s is not before start %s", asof, start)
		uploadOK = false
	}
	// should we check that all the x.Meta are consistent for GOOS, GOARCH, etc?
	report := &telemetry.Report{
		Config:   u.ConfigVersion,
		X:        computeRandom(), // json encodes all the bits
		Week:     expiryDate,
		LastWeek: lastWeek,
	}
	if report.X > u.Config.SampleRate && u.Config.SampleRate > 0 {
		logger.Printf("X:%f > SampleRate:%f, not uploadable", report.X, u.Config.SampleRate)
		uploadOK = false
	}
	var succeeded bool
	for _, f := range files {
		x, err := u.parse(string(f))
		if err != nil {
			logger.Printf("unparseable (%v) %s", err, f)
			continue
		}
		prog := findProgReport(x.Meta, report)
		for k, v := range x.Count {
			if counter.IsStackCounter(k) {
				// stack
				prog.Stacks[k] += int64(v)
			} else {
				// counter
				prog.Counters[k] += int64(v)
			}
			succeeded = true
		}
	}
	if !succeeded {
		return "", fmt.Errorf("all %d count files were unparseable", len(files))
	}
	// 1. generate the local report
	localContents, err := json.MarshalIndent(report, "", " ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal report (%v)", err)
	}
	// check that the report can be read back
	// TODO(pjw): remove for production?
	var x telemetry.Report
	if err := json.Unmarshal(localContents, &x); err != nil {
		return "", fmt.Errorf("failed to unmarshal local report (%v)", err)
	}

	var uploadContents []byte
	if uploadOK {
		// 2. create the uploadable version
		cfg := config.NewConfig(u.Config)
		upload := &telemetry.Report{
			Week:     report.Week,
			LastWeek: report.LastWeek,
			X:        report.X,
			Config:   report.Config,
		}
		for _, p := range report.Programs {
			// does the uploadConfig want this program?
			// if so, copy over the Stacks and Counters
			// that the uploadConfig mentions.
			if !cfg.HasGoVersion(p.GoVersion) || !cfg.HasProgram(p.Program) || !cfg.HasVersion(p.Program, p.Version) {
				continue
			}
			x := &telemetry.ProgramReport{
				Program:   p.Program,
				Version:   p.Version,
				GOOS:      p.GOOS,
				GOARCH:    p.GOARCH,
				GoVersion: p.GoVersion,
				Counters:  make(map[string]int64),
				Stacks:    make(map[string]int64),
			}
			upload.Programs = append(upload.Programs, x)
			for k, v := range p.Counters {
				if cfg.HasCounter(p.Program, k) && report.X <= cfg.Rate(p.Program, k) {
					x.Counters[k] = v
				}
			}
			// and the same for Stacks
			// this can be made more efficient, when it matters
			for k, v := range p.Stacks {
				before, _, _ := strings.Cut(k, "\n")
				if cfg.HasStack(p.Program, before) && report.X <= cfg.Rate(p.Program, before) {
					x.Stacks[k] = v
				}
			}
		}

		uploadContents, err = json.MarshalIndent(upload, "", " ")
		if err != nil {
			return "", fmt.Errorf("failed to marshal upload report (%v)", err)
		}
	}
	localFileName := filepath.Join(u.LocalDir, "local."+expiryDate+".json")
	uploadFileName := filepath.Join(u.LocalDir, expiryDate+".json")

	/* Prepare to write files */
	// if either file exists, someone has been here ahead of us
	// (there is still a race, but this check shortens the open window)
	if _, err := os.Stat(localFileName); err == nil {
		deleteFiles(files)
		return "", fmt.Errorf("local report %s already exists", localFileName)
	}
	if _, err := os.Stat(uploadFileName); err == nil {
		deleteFiles(files)
		return "", fmt.Errorf("report %s already exists", uploadFileName)
	}
	// write the uploadable file
	var errUpload, errLocal error
	if uploadOK {
		errUpload = os.WriteFile(uploadFileName, uploadContents, 0644)
	}
	// write the local file
	errLocal = os.WriteFile(localFileName, localContents, 0644)
	/*  Wrote the files */

	// even though these errors won't occur, what should happen
	// if errUpload == nil and it is ok to upload, and errLocal != nil?
	if errLocal != nil {
		return "", fmt.Errorf("failed to write local file %s (%v)", localFileName, errLocal)
	}
	if errUpload != nil {
		return "", fmt.Errorf("failed to write upload file %s (%v)", uploadFileName, errUpload)
	}
	logger.Printf("created %q, deleting %v", uploadFileName, files)
	deleteFiles(files)
	if uploadOK {
		return uploadFileName, nil
	}
	return "", nil
}

// return an existing ProgremReport, or create anew
func findProgReport(meta map[string]string, report *telemetry.Report) *telemetry.ProgramReport {
	for _, prog := range report.Programs {
		if prog.Program == meta["Program"] && prog.Version == meta["Version"] &&
			prog.GoVersion == meta["GoVersion"] && prog.GOOS == meta["GOOS"] &&
			prog.GOARCH == meta["GOARCH"] {
			return prog
		}
	}
	prog := telemetry.ProgramReport{
		Program:   meta["Program"],
		Version:   meta["Version"],
		GoVersion: meta["GoVersion"],
		GOOS:      meta["GOOS"],
		GOARCH:    meta["GOARCH"],
		Counters:  make(map[string]int64),
		Stacks:    make(map[string]int64),
	}
	report.Programs = append(report.Programs, &prog)
	return &prog
}

// turn 8 random bytes into a float64 in [0,1]
func computeRandom() float64 {
	for {
		b := make([]byte, 8)
		_, err := rand.Read(b)
		if err != nil {
			logger.Fatalf("rand.Read: %v", err)
		}
		// and turn it into a float64
		x := math.Float64frombits(binary.LittleEndian.Uint64(b))
		if math.IsNaN(x) || math.IsInf(x, 0) {
			continue
		}
		x = math.Abs(x)
		if x < 0x1p-1000 { // avoid underflow patterns
			continue
		}
		frac, _ := math.Frexp(x) // 52 bits of randomness
		return frac*2 - 1
	}
}
