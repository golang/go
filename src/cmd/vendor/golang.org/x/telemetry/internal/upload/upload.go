// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upload

import (
	"bytes"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var (
	dateRE     = regexp.MustCompile(`(\d\d\d\d-\d\d-\d\d)[.]json$`)
	dateFormat = time.DateOnly
	// TODO(rfindley): use dateFormat throughout.
)

// uploadReportDate returns the date component of the upload file name, or "" if the
// date was unmatched.
func (u *uploader) uploadReportDate(fname string) time.Time {
	match := dateRE.FindStringSubmatch(fname)
	if match == nil || len(match) < 2 {
		u.logger.Printf("malformed report name: missing date: %q", filepath.Base(fname))
		return time.Time{}
	}
	d, err := time.Parse(dateFormat, match[1])
	if err != nil {
		u.logger.Printf("malformed report name: bad date: %q", filepath.Base(fname))
		return time.Time{}
	}
	return d
}

func (u *uploader) uploadReport(fname string) {
	thisInstant := u.startTime
	// TODO(rfindley): use uploadReportDate here, once we've done a gopls release.

	// first make sure it is not in the future
	today := thisInstant.Format(time.DateOnly)
	match := dateRE.FindStringSubmatch(fname)
	if match == nil || len(match) < 2 {
		u.logger.Printf("Report name %q missing date", filepath.Base(fname))
	} else if match[1] > today {
		u.logger.Printf("Report date for %q is later than today (%s)", filepath.Base(fname), today)
		return // report is in the future, which shouldn't happen
	}
	buf, err := os.ReadFile(fname)
	if err != nil {
		u.logger.Printf("%v reading %s", err, fname)
		return
	}
	if u.uploadReportContents(fname, buf) {
		// anything left to do?
	}
}

// try to upload the report, 'true' if successful
func (u *uploader) uploadReportContents(fname string, buf []byte) bool {
	fdate := strings.TrimSuffix(filepath.Base(fname), ".json")
	fdate = fdate[len(fdate)-len(time.DateOnly):]

	newname := filepath.Join(u.dir.UploadDir(), fdate+".json")

	// Lock the upload, to prevent duplicate uploads.
	{
		lockname := newname + ".lock"
		lockfile, err := os.OpenFile(lockname, os.O_CREATE|os.O_EXCL, 0666)
		if err != nil {
			u.logger.Printf("Failed to acquire lock %s: %v", lockname, err)
			return false
		}
		_ = lockfile.Close()
		defer os.Remove(lockname)
	}

	if _, err := os.Stat(newname); err == nil {
		// Another process uploaded but failed to clean up (or hasn't yet cleaned
		// up). Ensure that cleanup occurs.
		u.logger.Printf("After acquire: report already uploaded")
		_ = os.Remove(fname)
		return false
	}

	endpoint := u.uploadServerURL + "/" + fdate
	b := bytes.NewReader(buf)
	resp, err := http.Post(endpoint, "application/json", b)
	if err != nil {
		u.logger.Printf("Error upload %s to %s: %v", filepath.Base(fname), endpoint, err)
		return false
	}
	// hope for a 200, remove file on a 4xx, otherwise it will be retried by another process
	if resp.StatusCode != 200 {
		u.logger.Printf("Failed to upload %s to %s: %s", filepath.Base(fname), endpoint, resp.Status)
		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			err := os.Remove(fname)
			if err == nil {
				u.logger.Printf("Removed local/%s", filepath.Base(fname))
			} else {
				u.logger.Printf("Error removing local/%s: %v", filepath.Base(fname), err)
			}
		}
		return false
	}
	// Store a copy of the uploaded report in the uploaded directory.
	if err := os.WriteFile(newname, buf, 0644); err == nil {
		os.Remove(fname) // if it exists
	}
	u.logger.Printf("Uploaded %s to %q", fdate+".json", endpoint)
	return true
}
