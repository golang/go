// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upload

import (
	"fmt"
	"os"
	"sync"
	"time"

	"golang.org/x/telemetry/internal/counter"
)

// time and date handling

var distantPast = 21 * 24 * time.Hour

// reports that are too old (21 days) are not uploaded
func (u *Uploader) tooOld(date string, uploadStartTime time.Time) bool {
	t, err := time.Parse("2006-01-02", date)
	if err != nil {
		u.logger.Printf("tooOld: %v", err)
		return false
	}
	age := uploadStartTime.Sub(t)
	return age > distantPast
}

// counterDateSpan parses the counter file named fname and returns the (begin,
// end) span recorded in its metadata, or an error if this data could not be
// extracted.
func (u *Uploader) counterDateSpan(fname string) (begin, end time.Time, _ error) {
	parsed, err := u.parseCountFile(fname)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	timeBegin, ok := parsed.Meta["TimeBegin"]
	if !ok {
		return time.Time{}, time.Time{}, fmt.Errorf("missing counter metadata for TimeBegin")
	}
	begin, err = time.Parse(time.RFC3339, timeBegin)
	if err != nil {
		return time.Time{}, time.Time{}, fmt.Errorf("failed to parse TimeBegin: %v", err)
	}
	timeEnd, ok := parsed.Meta["TimeEnd"]
	if !ok {
		return time.Time{}, time.Time{}, fmt.Errorf("missing counter metadata for TimeEnd")
	}
	end, err = time.Parse(time.RFC3339, timeEnd)
	if err != nil {
		return time.Time{}, time.Time{}, fmt.Errorf("failed to parse TimeEnd: %v", err)
	}
	return begin, end, nil
}

// avoid parsing count files multiple times
type parsedCache struct {
	mu sync.Mutex
	m  map[string]*counter.File
}

func (u *Uploader) parseCountFile(fname string) (*counter.File, error) {
	u.cache.mu.Lock()
	defer u.cache.mu.Unlock()
	if u.cache.m == nil {
		u.cache.m = make(map[string]*counter.File)
	}
	if f, ok := u.cache.m[fname]; ok {
		return f, nil
	}
	buf, err := os.ReadFile(fname)
	if err != nil {
		return nil, fmt.Errorf("parse ReadFile: %v for %s", err, fname)
	}
	f, err := counter.Parse(fname, buf)
	if err != nil {

		return nil, fmt.Errorf("parse Parse: %v for %s", err, fname)
	}
	u.cache.m[fname] = f
	return f, nil
}
