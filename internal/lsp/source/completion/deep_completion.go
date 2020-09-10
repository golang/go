// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"context"
	"go/types"
	"strings"
	"time"
)

// searchItem represents a candidate in deep completion search queue.
type searchItem struct {
	*searchPath
	cand candidate
}

// searchPath holds the path from search root (excluding the item itself) for
// a searchItem.
type searchPath struct {
	path  []types.Object
	names []string
}

// MaxDeepCompletions limits deep completion results because in most cases
// there are too many to be useful.
const MaxDeepCompletions = 3

// deepCompletionState stores our state as we search for deep completions.
// "deep completion" refers to searching into objects' fields and methods to
// find more completion candidates.
type deepCompletionState struct {
	// enabled indicates wether deep completion is permitted. It should be
	// reset to original value if manually disabled for an individual case.
	enabled bool

	// queueClosed is used to disable adding new items to search queue once
	// we're running out of our time budget.
	queueClosed bool

	// searchQueue holds the current breadth first search queue.
	searchQueue []searchItem

	// curPath tracks the current deep completion search path.
	curPath *searchPath

	// highScores tracks the highest deep candidate scores we have found
	// so far. This is used to avoid work for low scoring deep candidates.
	highScores [MaxDeepCompletions]float64

	// candidateCount is the count of unique deep candidates encountered
	// so far.
	candidateCount int
}

// enqueue adds candidates to the search queue.
func (s *deepCompletionState) enqueue(path *searchPath, candidates ...candidate) {
	for _, cand := range candidates {
		s.searchQueue = append(s.searchQueue, searchItem{path, cand})
	}
}

// dequeue removes and returns the leftmost element from the search queue.
func (s *deepCompletionState) dequeue() *searchItem {
	var item *searchItem
	item, s.searchQueue = &s.searchQueue[0], s.searchQueue[1:]
	return item
}

// scorePenalty computes a deep candidate score penalty. A candidate is
// penalized based on depth to favor shallower candidates. We also give a
// slight bonus to unexported objects and a slight additional penalty to
// function objects.
func (s *deepCompletionState) scorePenalty() float64 {
	var deepPenalty float64
	for _, dc := range s.curPath.path {
		deepPenalty++

		if !dc.Exported() {
			deepPenalty -= 0.1
		}

		if _, isSig := dc.Type().Underlying().(*types.Signature); isSig {
			deepPenalty += 0.1
		}
	}

	// Normalize penalty to a max depth of 10.
	return deepPenalty / 10
}

// isHighScore returns whether score is among the top MaxDeepCompletions deep
// candidate scores encountered so far. If so, it adds score to highScores,
// possibly displacing an existing high score.
func (s *deepCompletionState) isHighScore(score float64) bool {
	// Invariant: s.highScores is sorted with highest score first. Unclaimed
	// positions are trailing zeros.

	// If we beat an existing score then take its spot.
	for i, deepScore := range s.highScores {
		if score <= deepScore {
			continue
		}

		if deepScore != 0 && i != len(s.highScores)-1 {
			// If this wasn't an empty slot then we need to scooch everyone
			// down one spot.
			copy(s.highScores[i+1:], s.highScores[i:])
		}
		s.highScores[i] = score
		return true
	}

	return false
}

// inDeepCompletion returns if we're currently searching an object's members.
func (s *deepCompletionState) inDeepCompletion() bool {
	return len(s.curPath.path) > 0
}

// reset resets deepCompletionState since found might be called multiple times.
// We don't reset high scores since multiple calls to found still respect the
// same MaxDeepCompletions count.
func (s *deepCompletionState) reset() {
	s.searchQueue = nil
	s.curPath = &searchPath{}
}

// appendToSearchPath appends an object to a given searchPath.
func appendToSearchPath(oldPath searchPath, obj types.Object, invoke bool) *searchPath {
	name := obj.Name()
	if invoke {
		name += "()"
	}

	// copy the slice since we don't want to overwrite the original slice.
	path := append([]types.Object{}, oldPath.path...)
	names := append([]string{}, oldPath.names...)

	return &searchPath{
		path:  append(path, obj),
		names: append(names, name),
	}
}

// found adds a candidate to completion items if it's a valid suggestion and
// searches the candidate's subordinate objects for more completion items if
// deep completion is enabled.
func (c *completer) found(ctx context.Context, cand candidate) {
	// reset state at the end so current state doesn't affect completions done
	// outside c.found.
	defer c.deepState.reset()

	// At the top level, dedupe by object.
	if c.seen[cand.obj] {
		return
	}
	c.seen[cand.obj] = true

	c.deepState.enqueue(&searchPath{}, cand)
outer:
	for len(c.deepState.searchQueue) > 0 {
		item := c.deepState.dequeue()
		curCand := item.cand
		obj := curCand.obj

		// If obj is not accessible because it lives in another package and is
		// not exported, don't treat it as a completion candidate.
		if obj.Pkg() != nil && obj.Pkg() != c.pkg.GetTypes() && !obj.Exported() {
			continue
		}

		// If we want a type name, don't offer non-type name candidates.
		// However, do offer package names since they can contain type names,
		// and do offer any candidate without a type since we aren't sure if it
		// is a type name or not (i.e. unimported candidate).
		if c.wantTypeName() && obj.Type() != nil && !isTypeName(obj) && !isPkgName(obj) {
			continue
		}

		// When searching deep, make sure we don't have a cycle in our chain.
		// We don't dedupe by object because we want to allow both "foo.Baz"
		// and "bar.Baz" even though "Baz" is represented the same types.Object
		// in both.
		for _, seenObj := range item.path {
			if seenObj == obj {
				continue outer
			}
		}

		// update tracked current path since other functions might check it.
		c.deepState.curPath = item.searchPath

		c.addCandidate(ctx, curCand)

		c.deepState.candidateCount++
		if c.opts.budget > 0 && c.deepState.candidateCount%100 == 0 {
			spent := float64(time.Since(c.startTime)) / float64(c.opts.budget)
			if spent > 1.0 {
				return
			}
			// If we are almost out of budgeted time, no further elements
			// should be added to the queue. This ensures remaining time is
			// used for processing current queue.
			if !c.deepState.queueClosed && spent >= 0.85 {
				c.deepState.queueClosed = true
			}
		}

		// if deep search is disabled, don't add any more candidates.
		if !c.deepState.enabled || c.deepState.queueClosed {
			continue
		}

		// Searching members for a type name doesn't make sense.
		if isTypeName(obj) {
			continue
		}
		if obj.Type() == nil {
			continue
		}

		// Don't search embedded fields because they were already included in their
		// parent's fields.
		if v, ok := obj.(*types.Var); ok && v.Embedded() {
			continue
		}

		if sig, ok := obj.Type().Underlying().(*types.Signature); ok {
			// If obj is a function that takes no arguments and returns one
			// value, keep searching across the function call.
			if sig.Params().Len() == 0 && sig.Results().Len() == 1 {
				newSearchPath := appendToSearchPath(*item.searchPath, obj, true)
				// The result of a function call is not addressable.
				candidates := c.methodsAndFields(ctx, sig.Results().At(0).Type(), false, curCand.imp)
				c.deepState.enqueue(newSearchPath, candidates...)
			}
		}

		newSearchPath := appendToSearchPath(*item.searchPath, obj, false)
		switch obj := obj.(type) {
		case *types.PkgName:
			candidates := c.packageMembers(ctx, obj.Imported(), stdScore, curCand.imp)
			c.deepState.enqueue(newSearchPath, candidates...)
		default:
			candidates := c.methodsAndFields(ctx, obj.Type(), curCand.addressable, curCand.imp)
			c.deepState.enqueue(newSearchPath, candidates...)
		}
	}
}

// addCandidate adds a completion candidate to suggestions, without searching
// its members for more candidates.
func (c *completer) addCandidate(ctx context.Context, cand candidate) {
	obj := cand.obj
	if c.matchingCandidate(&cand) {
		cand.score *= highScore

		if p := c.penalty(&cand); p > 0 {
			cand.score *= (1 - p)
		}
	} else if isTypeName(obj) {
		// If obj is a *types.TypeName that didn't otherwise match, check
		// if a literal object of this type makes a good candidate.

		// We only care about named types (i.e. don't want builtin types).
		if _, isNamed := obj.Type().(*types.Named); isNamed {
			c.literal(ctx, obj.Type(), cand.imp)
		}
	}

	// Lower score of method calls so we prefer fields and vars over calls.
	if cand.expandFuncCall {
		if sig, ok := obj.Type().Underlying().(*types.Signature); ok && sig.Recv() != nil {
			cand.score *= 0.9
		}
	}

	// Prefer private objects over public ones.
	if !obj.Exported() && obj.Parent() != types.Universe {
		cand.score *= 1.1
	}

	// Favor shallow matches by lowering score according to depth.
	cand.score -= cand.score * c.deepState.scorePenalty()

	if cand.score < 0 {
		cand.score = 0
	}

	cand.name = strings.Join(append(c.deepState.curPath.names, cand.obj.Name()), ".")
	matchScore := c.matcher.Score(cand.name)
	if matchScore > 0 {
		cand.score *= float64(matchScore)

		// Avoid calling c.item() for deep candidates that wouldn't be in the top
		// MaxDeepCompletions anyway.
		if !c.deepState.inDeepCompletion() || c.deepState.isHighScore(cand.score) {
			if item, err := c.item(ctx, cand); err == nil {
				c.items = append(c.items, item)
			}
		}
	}

}

// penalty reports a score penalty for cand in the range (0, 1).
// For example, a candidate is penalized if it has already been used
// in another switch case statement.
func (c *completer) penalty(cand *candidate) float64 {
	for _, p := range c.inference.penalized {
		if c.objChainMatches(cand.obj, p.objChain) {
			return p.penalty
		}
	}

	return 0
}

// objChainMatches reports whether cand combined with the surrounding
// object prefix matches chain.
func (c *completer) objChainMatches(cand types.Object, chain []types.Object) bool {
	// For example, when completing:
	//
	//   foo.ba<>
	//
	// If we are considering the deep candidate "bar.baz", cand is baz,
	// objChain is [foo] and deepChain is [bar]. We would match the
	// chain [foo, bar, baz].

	if len(chain) != len(c.inference.objChain)+len(c.deepState.curPath.path)+1 {
		return false
	}

	if chain[len(chain)-1] != cand {
		return false
	}

	for i, o := range c.inference.objChain {
		if chain[i] != o {
			return false
		}
	}

	for i, o := range c.deepState.curPath.path {
		if chain[i+len(c.inference.objChain)] != o {
			return false
		}
	}

	return true
}
