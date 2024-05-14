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
	"golang.org/x/telemetry/internal/counter"
	"golang.org/x/telemetry/internal/telemetry"
)

// reports generates reports from inactive count files
func (u *Uploader) reports(todo *work) ([]string, error) {
	if mode, _ := u.dir.Mode(); mode == "off" {
		return nil, nil // no reports
	}
	thisInstant := u.startTime
	today := thisInstant.Format("2006-01-02")
	lastWeek := latestReport(todo.uploaded)
	if lastWeek >= today { //should never happen
		lastWeek = ""
	}
	u.logger.Printf("Last week: %s, today: %s", lastWeek, today)
	countFiles := make(map[string][]string) // expiry date string->filenames
	earliest := make(map[string]time.Time)  // earliest begin time for any counter
	for _, f := range todo.countfiles {
		begin, end, err := u.counterDateSpan(f)
		if err != nil {
			// This shouldn't happen: we should have already skipped count files that
			// don't contain valid start or end times.
			u.logger.Printf("BUG: failed to parse expiry for collected count file: %v", err)
			continue
		}

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
			u.logger.Printf("Files for %s not needed, deleting %v", expiry, files)
			// The report already exists.
			// There's another check in createReport.
			u.deleteFiles(files)
			continue
		}
		fname, err := u.createReport(earliest[expiry], expiry, files, lastWeek)
		if err != nil {
			u.logger.Printf("Failed to create report for %s: %v", expiry, err)
			continue
		}
		if fname != "" {
			u.logger.Printf("Ready to upload: %s", filepath.Base(fname))
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

func (u *Uploader) deleteFiles(files []string) {
	for _, f := range files {
		if err := os.Remove(f); err != nil {
			// this could be a race condition.
			// conversely, on Windows, err may be nil and
			// the file not deleted if anyone has it open.
			u.logger.Printf("%v failed to remove %s", err, f)
		}
	}
}

// createReport for all the count files for the same date.
// returns the absolute path name of the file containing the report
func (u *Uploader) createReport(start time.Time, expiryDate string, countFiles []string, lastWeek string) (string, error) {
	uploadOK := true
	mode, asof := u.dir.Mode()
	if mode != "on" {
		u.logger.Printf("No upload config or mode %q is not 'on'", mode)
		uploadOK = false // no config, nothing to upload
	}
	if u.tooOld(expiryDate, u.startTime) {
		u.logger.Printf("Expiry date %s is too old", expiryDate)
		uploadOK = false
	}
	// If the mode is recorded with an asof date, don't upload if the report
	// includes any data on or before the asof date.
	if !asof.IsZero() && !asof.Before(start) {
		u.logger.Printf("As-of date %s is not before start %s", asof, start)
		uploadOK = false
	}
	// TODO(rfindley): check that all the x.Meta are consistent for GOOS, GOARCH, etc.
	report := &telemetry.Report{
		Config:   u.configVersion,
		X:        computeRandom(), // json encodes all the bits
		Week:     expiryDate,
		LastWeek: lastWeek,
	}
	if report.X > u.config.SampleRate && u.config.SampleRate > 0 {
		u.logger.Printf("X: %f > SampleRate:%f, not uploadable", report.X, u.config.SampleRate)
		uploadOK = false
	}
	var succeeded bool
	for _, f := range countFiles {
		fok := false
		x, err := u.parseCountFile(f)
		if err != nil {
			u.logger.Printf("Unparseable count file %s: %v", filepath.Base(f), err)
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
			fok = true
		}
		if !fok {
			u.logger.Printf("no counters found in %s", f)
		}
	}
	if !succeeded {
		return "", fmt.Errorf("none of the %d count files for %s contained counters", len(countFiles), expiryDate)
	}
	// 1. generate the local report
	localContents, err := json.MarshalIndent(report, "", " ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal report for %s: %v", expiryDate, err)
	}
	// check that the report can be read back
	// TODO(pjw): remove for production?
	var report2 telemetry.Report
	if err := json.Unmarshal(localContents, &report2); err != nil {
		return "", fmt.Errorf("failed to unmarshal local report for %s: %v", expiryDate, err)
	}

	var uploadContents []byte
	if uploadOK {
		// 2. create the uploadable version
		cfg := config.NewConfig(u.config)
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
			return "", fmt.Errorf("failed to marshal upload report for %s: %v", expiryDate, err)
		}
	}
	localFileName := filepath.Join(u.dir.LocalDir(), "local."+expiryDate+".json")
	uploadFileName := filepath.Join(u.dir.LocalDir(), expiryDate+".json")

	/* Prepare to write files */
	// if either file exists, someone has been here ahead of us
	// (there is still a race, but this check shortens the open window)
	if _, err := os.Stat(localFileName); err == nil {
		u.deleteFiles(countFiles)
		return "", fmt.Errorf("local report %s already exists", localFileName)
	}
	if _, err := os.Stat(uploadFileName); err == nil {
		u.deleteFiles(countFiles)
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
	u.logger.Printf("Created %s, deleting %d count files", filepath.Base(uploadFileName), len(countFiles))
	u.deleteFiles(countFiles)
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

// computeRandom returns a cryptographic random float64 in the range [0, 1],
// with 52 bits of precision.
func computeRandom() float64 {
	for {
		b := make([]byte, 8)
		_, err := rand.Read(b)
		if err != nil {
			panic(fmt.Sprintf("rand.Read failed: %v", err))
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
