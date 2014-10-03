// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implements methods to remove frames from profiles.

package profile

import (
	"fmt"
	"regexp"
)

// Prune removes all nodes beneath a node matching dropRx, and not
// matching keepRx.  If the root node of a Sample matches, the sample
// will have an empty stack.
func (p *Profile) Prune(dropRx, keepRx *regexp.Regexp) {
	prune := make(map[uint64]bool)
	pruneBeneath := make(map[uint64]bool)

	for _, loc := range p.Location {
		var i int
		for i = len(loc.Line) - 1; i >= 0; i-- {
			if fn := loc.Line[i].Function; fn != nil && fn.Name != "" {
				funcName := fn.Name
				// Account for leading '.' on the PPC ELF v1 ABI.
				if funcName[0] == '.' {
					funcName = funcName[1:]
				}
				if dropRx.MatchString(funcName) {
					if keepRx == nil || !keepRx.MatchString(funcName) {
						break
					}
				}
			}
		}

		if i >= 0 {
			// Found matching entry to prune.
			pruneBeneath[loc.ID] = true

			// Remove the matching location.
			if i == len(loc.Line)-1 {
				// Matched the top entry: prune the whole location.
				prune[loc.ID] = true
			} else {
				loc.Line = loc.Line[i+1:]
			}
		}
	}

	// Prune locs from each Sample
	for _, sample := range p.Sample {
		// Scan from the root to the leaves to find the prune location.
		// Do not prune frames before the first user frame, to avoid
		// pruning everything.
		foundUser := false
		for i := len(sample.Location) - 1; i >= 0; i-- {
			id := sample.Location[i].ID
			if !prune[id] && !pruneBeneath[id] {
				foundUser = true
				continue
			}
			if !foundUser {
				continue
			}
			if prune[id] {
				sample.Location = sample.Location[i+1:]
				break
			}
			if pruneBeneath[id] {
				sample.Location = sample.Location[i:]
				break
			}
		}
	}
}

// RemoveUninteresting prunes and elides profiles using built-in
// tables of uninteresting function names.
func (p *Profile) RemoveUninteresting() error {
	var keep, drop *regexp.Regexp
	var err error

	if p.DropFrames != "" {
		if drop, err = regexp.Compile("^(" + p.DropFrames + ")$"); err != nil {
			return fmt.Errorf("failed to compile regexp %s: %v", p.DropFrames, err)
		}
		if p.KeepFrames != "" {
			if keep, err = regexp.Compile("^(" + p.KeepFrames + ")$"); err != nil {
				return fmt.Errorf("failed to compile regexp %s: %v", p.KeepFrames, err)
			}
		}
		p.Prune(drop, keep)
	}
	return nil
}
