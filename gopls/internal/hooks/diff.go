// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hooks

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/sergi/go-diff/diffmatchpatch"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/diff"
)

// structure for saving information about diffs
// while the new code is being rolled out
type diffstat struct {
	Before, After      int
	Oldedits, Newedits int
	Oldtime, Newtime   time.Duration
	Stack              string
	Msg                string `json:",omitempty"` // for errors
	Ignored            int    `json:",omitempty"` // numbr of skipped records with 0 edits
}

var (
	ignoredMu sync.Mutex
	ignored   int // counter of diff requests on equal strings

	diffStatsOnce sync.Once
	diffStats     *os.File // never closed
)

// save writes a JSON record of statistics about diff requests to a temporary file.
func (s *diffstat) save() {
	diffStatsOnce.Do(func() {
		f, err := ioutil.TempFile("", "gopls-diff-stats-*")
		if err != nil {
			log.Printf("can't create diff stats temp file: %v", err) // e.g. disk full
			return
		}
		diffStats = f
	})
	if diffStats == nil {
		return
	}

	// diff is frequently called with equal strings,
	// so we count repeated instances but only print every 15th.
	ignoredMu.Lock()
	if s.Oldedits == 0 && s.Newedits == 0 {
		ignored++
		if ignored < 15 {
			ignoredMu.Unlock()
			return
		}
	}
	s.Ignored = ignored
	ignored = 0
	ignoredMu.Unlock()

	// Record the name of the file in which diff was called.
	// There aren't many calls, so only the base name is needed.
	if _, file, line, ok := runtime.Caller(2); ok {
		s.Stack = fmt.Sprintf("%s:%d", filepath.Base(file), line)
	}
	x, err := json.Marshal(s)
	if err != nil {
		log.Fatalf("internal error marshalling JSON: %v", err)
	}
	fmt.Fprintf(diffStats, "%s\n", x)
}

// disaster is called when the diff algorithm panics or produces a
// diff that cannot be applied. It saves the broken input in a
// new temporary file and logs the file name, which is returned.
func disaster(before, after string) string {
	// We use the pid to salt the name, not os.TempFile,
	// so that each process creates at most one file.
	// One is sufficient for a bug report.
	filename := fmt.Sprintf("%s/gopls-diff-bug-%x", os.TempDir(), os.Getpid())

	// We use NUL as a separator: it should never appear in Go source.
	data := before + "\x00" + after

	if err := ioutil.WriteFile(filename, []byte(data), 0600); err != nil {
		log.Printf("failed to write diff bug report: %v", err)
		return ""
	}

	// TODO(adonovan): is there a better way to surface this?
	log.Printf("Bug detected in diff algorithm! Please send file %s to the maintainers of gopls if you are comfortable sharing its contents.", filename)

	return filename
}

// BothDiffs edits calls both the new and old diffs, checks that the new diffs
// change before into after, and attempts to preserve some statistics.
func BothDiffs(before, after string) (edits []diff.Edit) {
	// The new diff code contains a lot of internal checks that panic when they
	// fail. This code catches the panics, or other failures, tries to save
	// the failing example (and it would ask the user to send it back to us, and
	// changes options.newDiff to 'old', if only we could figure out how.)
	stat := diffstat{Before: len(before), After: len(after)}
	now := time.Now()
	oldedits := ComputeEdits(before, after)
	stat.Oldedits = len(oldedits)
	stat.Oldtime = time.Since(now)
	defer func() {
		if r := recover(); r != nil {
			disaster(before, after)
			edits = oldedits
		}
	}()
	now = time.Now()
	newedits := diff.Strings(before, after)
	stat.Newedits = len(newedits)
	stat.Newtime = time.Now().Sub(now)
	got, err := diff.Apply(before, newedits)
	if err != nil || got != after {
		stat.Msg += "FAIL"
		disaster(before, after)
		stat.save()
		return oldedits
	}
	stat.save()
	return newedits
}

// ComputeEdits computes a diff using the github.com/sergi/go-diff implementation.
func ComputeEdits(before, after string) (edits []diff.Edit) {
	// The go-diff library has an unresolved panic (see golang/go#278774).
	// TODO(rstambler): Remove the recover once the issue has been fixed
	// upstream.
	defer func() {
		if r := recover(); r != nil {
			bug.Reportf("unable to compute edits: %s", r)
			// Report one big edit for the whole file.
			edits = []diff.Edit{{
				Start: 0,
				End:   len(before),
				New:   after,
			}}
		}
	}()
	diffs := diffmatchpatch.New().DiffMain(before, after, true)
	edits = make([]diff.Edit, 0, len(diffs))
	offset := 0
	for _, d := range diffs {
		start := offset
		switch d.Type {
		case diffmatchpatch.DiffDelete:
			offset += len(d.Text)
			edits = append(edits, diff.Edit{Start: start, End: offset})
		case diffmatchpatch.DiffEqual:
			offset += len(d.Text)
		case diffmatchpatch.DiffInsert:
			edits = append(edits, diff.Edit{Start: start, End: start, New: d.Text})
		}
	}
	return edits
}
