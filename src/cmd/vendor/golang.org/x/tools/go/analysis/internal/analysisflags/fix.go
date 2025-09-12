// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysisflags

// This file defines the -fix logic common to unitchecker and
// {single,multi}checker.

import (
	"fmt"
	"go/format"
	"go/token"
	"log"
	"maps"
	"os"
	"sort"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/diff"
)

// FixAction abstracts a checker action (running one analyzer on one
// package) for the purposes of applying its diagnostics' fixes.
type FixAction struct {
	Name         string // e.g. "analyzer@package"
	FileSet      *token.FileSet
	ReadFileFunc analysisinternal.ReadFileFunc
	Diagnostics  []analysis.Diagnostic
}

// ApplyFixes attempts to apply the first suggested fix associated
// with each diagnostic reported by the specified actions.
// All fixes must have been validated by [analysisinternal.ValidateFixes].
//
// Each fix is treated as an independent change; fixes are merged in
// an arbitrary deterministic order as if by a three-way diff tool
// such as the UNIX diff3 command or 'git merge'. Any fix that cannot be
// cleanly merged is discarded, in which case the final summary tells
// the user to re-run the tool.
// TODO(adonovan): make the checker tool re-run the analysis itself.
//
// When the same file is analyzed as a member of both a primary
// package "p" and a test-augmented package "p [p.test]", there may be
// duplicate diagnostics and fixes. One set of fixes will be applied
// and the other will be discarded; but re-running the tool may then
// show zero fixes, which may cause the confused user to wonder what
// happened to the other ones.
// TODO(adonovan): consider pre-filtering completely identical fixes.
//
// A common reason for overlapping fixes is duplicate additions of the
// same import. The merge algorithm may often cleanly resolve such
// fixes, coalescing identical edits, but the merge may sometimes be
// confused by nearby changes.
//
// Even when merging succeeds, there is no guarantee that the
// composition of the two fixes is semantically correct. Coalescing
// identical edits is appropriate for imports, but not for, say,
// increments to a counter variable; the correct resolution in that
// case might be to increment it twice. Or consider two fixes that
// each delete the penultimate reference to an import or local
// variable: each fix is sound individually, and they may be textually
// distant from each other, but when both are applied, the program is
// no longer valid because it has an unreferenced import or local
// variable.
// TODO(adonovan): investigate replacing the final "gofmt" step with a
// formatter that applies the unused-import deletion logic of
// "goimports".
//
// Merging depends on both the order of fixes and they order of edits
// within them. For example, if three fixes add import "a" twice and
// import "b" once, the two imports of "a" may be combined if they
// appear in order [a, a, b], or not if they appear as [a, b, a].
// TODO(adonovan): investigate an algebraic approach to imports;
// that is, for fixes to Go source files, convert changes within the
// import(...) portion of the file into semantic edits, compose those
// edits algebraically, then convert the result back to edits.
//
// applyFixes returns success if all fixes are valid, could be cleanly
// merged, and the corresponding files were successfully updated.
//
// If the -diff flag was set, instead of updating the files it display the final
// patch composed of all the cleanly merged fixes.
//
// TODO(adonovan): handle file-system level aliases such as symbolic
// links using robustio.FileID.
func ApplyFixes(actions []FixAction, verbose bool) error {
	// Select fixes to apply.
	//
	// If there are several for a given Diagnostic, choose the first.
	// Preserve the order of iteration, for determinism.
	type fixact struct {
		fix *analysis.SuggestedFix
		act FixAction
	}
	var fixes []*fixact
	for _, act := range actions {
		for _, diag := range act.Diagnostics {
			for i := range diag.SuggestedFixes {
				fix := &diag.SuggestedFixes[i]
				if i == 0 {
					fixes = append(fixes, &fixact{fix, act})
				} else {
					// TODO(adonovan): abstract the logger.
					log.Printf("%s: ignoring alternative fix %q", act.Name, fix.Message)
				}
			}
		}
	}

	// Read file content on demand, from the virtual
	// file system that fed the analyzer (see #62292).
	//
	// This cache assumes that all successful reads for the same
	// file name return the same content.
	// (It is tempting to group fixes by package and do the
	// merge/apply/format steps one package at a time, but
	// packages are not disjoint, due to test variants, so this
	// would not really address the issue.)
	baselineContent := make(map[string][]byte)
	getBaseline := func(readFile analysisinternal.ReadFileFunc, filename string) ([]byte, error) {
		content, ok := baselineContent[filename]
		if !ok {
			var err error
			content, err = readFile(filename)
			if err != nil {
				return nil, err
			}
			baselineContent[filename] = content
		}
		return content, nil
	}

	// Apply each fix, updating the current state
	// only if the entire fix can be cleanly merged.
	accumulatedEdits := make(map[string][]diff.Edit)
	goodFixes := 0
fixloop:
	for _, fixact := range fixes {
		// Convert analysis.TextEdits to diff.Edits, grouped by file.
		// Precondition: a prior call to validateFix succeeded.
		fileEdits := make(map[string][]diff.Edit)
		for _, edit := range fixact.fix.TextEdits {
			file := fixact.act.FileSet.File(edit.Pos)

			baseline, err := getBaseline(fixact.act.ReadFileFunc, file.Name())
			if err != nil {
				log.Printf("skipping fix to file %s: %v", file.Name(), err)
				continue fixloop
			}

			// We choose to treat size mismatch as a serious error,
			// as it indicates a concurrent write to at least one file,
			// and possibly others (consider a git checkout, for example).
			if file.Size() != len(baseline) {
				return fmt.Errorf("concurrent file modification detected in file %s (size changed from %d -> %d bytes); aborting fix",
					file.Name(), file.Size(), len(baseline))
			}

			fileEdits[file.Name()] = append(fileEdits[file.Name()], diff.Edit{
				Start: file.Offset(edit.Pos),
				End:   file.Offset(edit.End),
				New:   string(edit.NewText),
			})
		}

		// Apply each set of edits by merging atop
		// the previous accumulated state.
		after := make(map[string][]diff.Edit)
		for file, edits := range fileEdits {
			if prev := accumulatedEdits[file]; len(prev) > 0 {
				merged, ok := diff.Merge(prev, edits)
				if !ok {
					// debugging
					if false {
						log.Printf("%s: fix %s conflicts", fixact.act.Name, fixact.fix.Message)
					}
					continue fixloop // conflict
				}
				edits = merged
			}
			after[file] = edits
		}

		// The entire fix applied cleanly; commit it.
		goodFixes++
		maps.Copy(accumulatedEdits, after)
		// debugging
		if false {
			log.Printf("%s: fix %s applied", fixact.act.Name, fixact.fix.Message)
		}
	}
	badFixes := len(fixes) - goodFixes

	// Show diff or update files to final state.
	var files []string
	for file := range accumulatedEdits {
		files = append(files, file)
	}
	sort.Strings(files) // for deterministic -diff
	var filesUpdated, totalFiles int
	for _, file := range files {
		edits := accumulatedEdits[file]
		if len(edits) == 0 {
			continue // the diffs annihilated (a miracle?)
		}

		// Apply accumulated fixes.
		baseline := baselineContent[file] // (cache hit)
		final, err := diff.ApplyBytes(baseline, edits)
		if err != nil {
			log.Fatalf("internal error in diff.ApplyBytes: %v", err)
		}

		// Attempt to format each file.
		if formatted, err := format.Source(final); err == nil {
			final = formatted
		}

		if diffFlag {
			// Since we formatted the file, we need to recompute the diff.
			unified := diff.Unified(file+" (old)", file+" (new)", string(baseline), string(final))
			// TODO(adonovan): abstract the I/O.
			os.Stdout.WriteString(unified)

		} else {
			// write
			totalFiles++
			// TODO(adonovan): abstract the I/O.
			if err := os.WriteFile(file, final, 0644); err != nil {
				log.Println(err)
				continue
			}
			filesUpdated++
		}
	}

	// TODO(adonovan): consider returning a structured result that
	// maps each SuggestedFix to its status:
	// - invalid
	// - secondary, not selected
	// - applied
	// - had conflicts.
	// and a mapping from each affected file to:
	// - its final/original content pair, and
	// - whether formatting was successful.
	// Then file writes and the UI can be applied by the caller
	// in whatever form they like.

	// If victory was incomplete, report an error that indicates partial progress.
	//
	// badFixes > 0 indicates that we decided not to attempt some
	// fixes due to conflicts or failure to read the source; still
	// it's a relatively benign situation since the user can
	// re-run the tool, and we may still make progress.
	//
	// filesUpdated < totalFiles indicates that some file updates
	// failed. This should be rare, but is a serious error as it
	// may apply half a fix, or leave the files in a bad state.
	//
	// These numbers are potentially misleading:
	// The denominator includes duplicate conflicting fixes due to
	// common files in packages "p" and "p [p.test]", which may
	// have been fixed fixed and won't appear in the re-run.
	// TODO(adonovan): eliminate identical fixes as an initial
	// filtering step.
	//
	// TODO(adonovan): should we log that n files were updated in case of total victory?
	if badFixes > 0 || filesUpdated < totalFiles {
		if diffFlag {
			return fmt.Errorf("%d of %d fixes skipped (e.g. due to conflicts)", badFixes, len(fixes))
		} else {
			return fmt.Errorf("applied %d of %d fixes; %d files updated. (Re-run the command to apply more.)",
				goodFixes, len(fixes), filesUpdated)
		}
	}

	if verbose {
		log.Printf("applied %d fixes, updated %d files", len(fixes), filesUpdated)
	}

	return nil
}
