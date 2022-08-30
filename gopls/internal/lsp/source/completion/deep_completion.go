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

// MaxDeepCompletions limits deep completion results because in most cases
// there are too many to be useful.
const MaxDeepCompletions = 3

// deepCompletionState stores our state as we search for deep completions.
// "deep completion" refers to searching into objects' fields and methods to
// find more completion candidates.
type deepCompletionState struct {
	// enabled indicates whether deep completion is permitted.
	enabled bool

	// queueClosed is used to disable adding new sub-fields to search queue
	// once we're running out of our time budget.
	queueClosed bool

	// thisQueue holds the current breadth first search queue.
	thisQueue []candidate

	// nextQueue holds the next breadth first search iteration's queue.
	nextQueue []candidate

	// highScores tracks the highest deep candidate scores we have found
	// so far. This is used to avoid work for low scoring deep candidates.
	highScores [MaxDeepCompletions]float64

	// candidateCount is the count of unique deep candidates encountered
	// so far.
	candidateCount int
}

// enqueue adds a candidate to the search queue.
func (s *deepCompletionState) enqueue(cand candidate) {
	s.nextQueue = append(s.nextQueue, cand)
}

// dequeue removes and returns the leftmost element from the search queue.
func (s *deepCompletionState) dequeue() *candidate {
	var cand *candidate
	cand, s.thisQueue = &s.thisQueue[len(s.thisQueue)-1], s.thisQueue[:len(s.thisQueue)-1]
	return cand
}

// scorePenalty computes a deep candidate score penalty. A candidate is
// penalized based on depth to favor shallower candidates. We also give a
// slight bonus to unexported objects and a slight additional penalty to
// function objects.
func (s *deepCompletionState) scorePenalty(cand *candidate) float64 {
	var deepPenalty float64
	for _, dc := range cand.path {
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

// newPath returns path from search root for an object following a given
// candidate.
func (s *deepCompletionState) newPath(cand candidate, obj types.Object) []types.Object {
	path := make([]types.Object, len(cand.path)+1)
	copy(path, cand.path)
	path[len(path)-1] = obj

	return path
}

// deepSearch searches a candidate and its subordinate objects for completion
// items if deep completion is enabled and adds the valid candidates to
// completion items.
func (c *completer) deepSearch(ctx context.Context) {
	defer func() {
		// We can return early before completing the search, so be sure to
		// clear out our queues to not impact any further invocations.
		c.deepState.thisQueue = c.deepState.thisQueue[:0]
		c.deepState.nextQueue = c.deepState.nextQueue[:0]
	}()

	for len(c.deepState.nextQueue) > 0 {
		c.deepState.thisQueue, c.deepState.nextQueue = c.deepState.nextQueue, c.deepState.thisQueue[:0]

	outer:
		for _, cand := range c.deepState.thisQueue {
			obj := cand.obj

			if obj == nil {
				continue
			}

			// At the top level, dedupe by object.
			if len(cand.path) == 0 {
				if c.seen[obj] {
					continue
				}
				c.seen[obj] = true
			}

			// If obj is not accessible because it lives in another package and is
			// not exported, don't treat it as a completion candidate unless it's
			// a package completion candidate.
			if !c.completionContext.packageCompletion &&
				obj.Pkg() != nil && obj.Pkg() != c.pkg.GetTypes() && !obj.Exported() {
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
			for _, seenObj := range cand.path {
				if seenObj == obj {
					continue outer
				}
			}

			c.addCandidate(ctx, &cand)

			c.deepState.candidateCount++
			if c.opts.budget > 0 && c.deepState.candidateCount%100 == 0 {
				spent := float64(time.Since(c.startTime)) / float64(c.opts.budget)
				select {
				case <-ctx.Done():
					return
				default:
					// If we are almost out of budgeted time, no further elements
					// should be added to the queue. This ensures remaining time is
					// used for processing current queue.
					if !c.deepState.queueClosed && spent >= 0.85 {
						c.deepState.queueClosed = true
					}
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
					path := c.deepState.newPath(cand, obj)
					// The result of a function call is not addressable.
					c.methodsAndFields(sig.Results().At(0).Type(), false, cand.imp, func(newCand candidate) {
						newCand.pathInvokeMask = cand.pathInvokeMask | (1 << uint64(len(cand.path)))
						newCand.path = path
						c.deepState.enqueue(newCand)
					})
				}
			}

			path := c.deepState.newPath(cand, obj)
			switch obj := obj.(type) {
			case *types.PkgName:
				c.packageMembers(obj.Imported(), stdScore, cand.imp, func(newCand candidate) {
					newCand.pathInvokeMask = cand.pathInvokeMask
					newCand.path = path
					c.deepState.enqueue(newCand)
				})
			default:
				c.methodsAndFields(obj.Type(), cand.addressable, cand.imp, func(newCand candidate) {
					newCand.pathInvokeMask = cand.pathInvokeMask
					newCand.path = path
					c.deepState.enqueue(newCand)
				})
			}
		}
	}
}

// addCandidate adds a completion candidate to suggestions, without searching
// its members for more candidates.
func (c *completer) addCandidate(ctx context.Context, cand *candidate) {
	obj := cand.obj
	if c.matchingCandidate(cand) {
		cand.score *= highScore

		if p := c.penalty(cand); p > 0 {
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
	if cand.hasMod(invoke) {
		if sig, ok := obj.Type().Underlying().(*types.Signature); ok && sig.Recv() != nil {
			cand.score *= 0.9
		}
	}

	// Prefer private objects over public ones.
	if !obj.Exported() && obj.Parent() != types.Universe {
		cand.score *= 1.1
	}

	// Slight penalty for index modifier (e.g. changing "foo" to
	// "foo[]") to curb false positives.
	if cand.hasMod(index) {
		cand.score *= 0.9
	}

	// Favor shallow matches by lowering score according to depth.
	cand.score -= cand.score * c.deepState.scorePenalty(cand)

	if cand.score < 0 {
		cand.score = 0
	}

	cand.name = deepCandName(cand)
	if item, err := c.item(ctx, *cand); err == nil {
		c.items = append(c.items, item)
	}
}

// deepCandName produces the full candidate name including any
// ancestor objects. For example, "foo.bar().baz" for candidate "baz".
func deepCandName(cand *candidate) string {
	totalLen := len(cand.obj.Name())
	for i, obj := range cand.path {
		totalLen += len(obj.Name()) + 1
		if cand.pathInvokeMask&(1<<uint16(i)) > 0 {
			totalLen += 2
		}
	}

	var buf strings.Builder
	buf.Grow(totalLen)

	for i, obj := range cand.path {
		buf.WriteString(obj.Name())
		if cand.pathInvokeMask&(1<<uint16(i)) > 0 {
			buf.WriteByte('(')
			buf.WriteByte(')')
		}
		buf.WriteByte('.')
	}

	buf.WriteString(cand.obj.Name())

	return buf.String()
}

// penalty reports a score penalty for cand in the range (0, 1).
// For example, a candidate is penalized if it has already been used
// in another switch case statement.
func (c *completer) penalty(cand *candidate) float64 {
	for _, p := range c.inference.penalized {
		if c.objChainMatches(cand, p.objChain) {
			return p.penalty
		}
	}

	return 0
}

// objChainMatches reports whether cand combined with the surrounding
// object prefix matches chain.
func (c *completer) objChainMatches(cand *candidate, chain []types.Object) bool {
	// For example, when completing:
	//
	//   foo.ba<>
	//
	// If we are considering the deep candidate "bar.baz", cand is baz,
	// objChain is [foo] and deepChain is [bar]. We would match the
	// chain [foo, bar, baz].
	if len(chain) != len(c.inference.objChain)+len(cand.path)+1 {
		return false
	}

	if chain[len(chain)-1] != cand.obj {
		return false
	}

	for i, o := range c.inference.objChain {
		if chain[i] != o {
			return false
		}
	}

	for i, o := range cand.path {
		if chain[i+len(c.inference.objChain)] != o {
			return false
		}
	}

	return true
}
