// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Implements methods to remove frames from profiles.

package profile

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	reservedNames = []string{"(anonymous namespace)", "operator()"}
	bracketRx     = func() *regexp.Regexp {
		var quotedNames []string
		for _, name := range append(reservedNames, "(") {
			quotedNames = append(quotedNames, regexp.QuoteMeta(name))
		}
		return regexp.MustCompile(strings.Join(quotedNames, "|"))
	}()
)

// simplifyFunc does some primitive simplification of function names.
func simplifyFunc(f string) string {
	// Account for leading '.' on the PPC ELF v1 ABI.
	funcName := strings.TrimPrefix(f, ".")
	// Account for unsimplified names -- try  to remove the argument list by trimming
	// starting from the first '(', but skipping reserved names that have '('.
	for _, ind := range bracketRx.FindAllStringSubmatchIndex(funcName, -1) {
		foundReserved := false
		for _, res := range reservedNames {
			if funcName[ind[0]:ind[1]] == res {
				foundReserved = true
				break
			}
		}
		if !foundReserved {
			funcName = funcName[:ind[0]]
			break
		}
	}
	return funcName
}

// Prune removes all nodes beneath a node matching dropRx, and not
// matching keepRx. If the root node of a Sample matches, the sample
// will have an empty stack.
func (p *Profile) Prune(dropRx, keepRx *regexp.Regexp) {
	prune := make(map[uint64]bool)
	pruneBeneath := make(map[uint64]bool)

	// simplifyFunc can be expensive, so cache results.
	// Note that the same function name can be encountered many times due
	// different lines and addresses in the same function.
	pruneCache := map[string]bool{} // Map from function to whether or not to prune
	pruneFromHere := func(s string) bool {
		if r, ok := pruneCache[s]; ok {
			return r
		}
		funcName := simplifyFunc(s)
		if dropRx.MatchString(funcName) {
			if keepRx == nil || !keepRx.MatchString(funcName) {
				pruneCache[s] = true
				return true
			}
		}
		pruneCache[s] = false
		return false
	}

	for _, loc := range p.Location {
		var i int
		for i = len(loc.Line) - 1; i >= 0; i-- {
			if fn := loc.Line[i].Function; fn != nil && fn.Name != "" {
				if pruneFromHere(fn.Name) {
					break
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

// PruneFrom removes all nodes beneath the lowest node matching dropRx, not including itself.
//
// Please see the example below to understand this method as well as
// the difference from Prune method.
//
// A sample contains Location of [A,B,C,B,D] where D is the top frame and there's no inline.
//
// PruneFrom(A) returns [A,B,C,B,D] because there's no node beneath A.
// Prune(A, nil) returns [B,C,B,D] by removing A itself.
//
// PruneFrom(B) returns [B,C,B,D] by removing all nodes beneath the first B when scanning from the bottom.
// Prune(B, nil) returns [D] because a matching node is found by scanning from the root.
func (p *Profile) PruneFrom(dropRx *regexp.Regexp) {
	pruneBeneath := make(map[uint64]bool)

	for _, loc := range p.Location {
		for i := 0; i < len(loc.Line); i++ {
			if fn := loc.Line[i].Function; fn != nil && fn.Name != "" {
				funcName := simplifyFunc(fn.Name)
				if dropRx.MatchString(funcName) {
					// Found matching entry to prune.
					pruneBeneath[loc.ID] = true
					loc.Line = loc.Line[i:]
					break
				}
			}
		}
	}

	// Prune locs from each Sample
	for _, sample := range p.Sample {
		// Scan from the bottom leaf to the root to find the prune location.
		for i, loc := range sample.Location {
			if pruneBeneath[loc.ID] {
				sample.Location = sample.Location[i:]
				break
			}
		}
	}
}
