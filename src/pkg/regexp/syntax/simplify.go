// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// Simplify returns a regexp equivalent to re but without counted repetitions
// and with various other simplifications, such as rewriting /(?:a+)+/ to /a+/.
// The resulting regexp will execute correctly but its string representation
// will not produce the same parse tree, because capturing parentheses
// may have been duplicated or removed.  For example, the simplified form
// for /(x){1,2}/ is /(x)(x)?/ but both parentheses capture as $1.
// The returned regexp may share structure with or be the original.
func (re *Regexp) Simplify() *Regexp {
	if re == nil {
		return nil
	}
	switch re.Op {
	case OpCapture, OpConcat, OpAlternate:
		// Simplify children, building new Regexp if children change.
		nre := re
		for i, sub := range re.Sub {
			nsub := sub.Simplify()
			if nre == re && nsub != sub {
				// Start a copy.
				nre = new(Regexp)
				*nre = *re
				nre.Rune = nil
				nre.Sub = append(nre.Sub0[:0], re.Sub[:i]...)
			}
			if nre != re {
				nre.Sub = append(nre.Sub, nsub)
			}
		}
		return nre

	case OpStar, OpPlus, OpQuest:
		sub := re.Sub[0].Simplify()
		return simplify1(re.Op, re.Flags, sub, re)

	case OpRepeat:
		// Special special case: x{0} matches the empty string
		// and doesn't even need to consider x.
		if re.Min == 0 && re.Max == 0 {
			return &Regexp{Op: OpEmptyMatch}
		}

		// The fun begins.
		sub := re.Sub[0].Simplify()

		// x{n,} means at least n matches of x.
		if re.Max == -1 {
			// Special case: x{0,} is x*.
			if re.Min == 0 {
				return simplify1(OpStar, re.Flags, sub, nil)
			}

			// Special case: x{1,} is x+.
			if re.Min == 1 {
				return simplify1(OpPlus, re.Flags, sub, nil)
			}

			// General case: x{4,} is xxxx+.
			nre := &Regexp{Op: OpConcat}
			nre.Sub = nre.Sub0[:0]
			for i := 0; i < re.Min-1; i++ {
				nre.Sub = append(nre.Sub, sub)
			}
			nre.Sub = append(nre.Sub, simplify1(OpPlus, re.Flags, sub, nil))
			return nre
		}

		// Special case x{0} handled above.

		// Special case: x{1} is just x.
		if re.Min == 1 && re.Max == 1 {
			return sub
		}

		// General case: x{n,m} means n copies of x and m copies of x?
		// The machine will do less work if we nest the final m copies,
		// so that x{2,5} = xx(x(x(x)?)?)?

		// Build leading prefix: xx.
		var prefix *Regexp
		if re.Min > 0 {
			prefix = &Regexp{Op: OpConcat}
			prefix.Sub = prefix.Sub0[:0]
			for i := 0; i < re.Min; i++ {
				prefix.Sub = append(prefix.Sub, sub)
			}
		}

		// Build and attach suffix: (x(x(x)?)?)?
		if re.Max > re.Min {
			suffix := simplify1(OpQuest, re.Flags, sub, nil)
			for i := re.Min + 1; i < re.Max; i++ {
				nre2 := &Regexp{Op: OpConcat}
				nre2.Sub = append(nre2.Sub0[:0], sub, suffix)
				suffix = simplify1(OpQuest, re.Flags, nre2, nil)
			}
			if prefix == nil {
				return suffix
			}
			prefix.Sub = append(prefix.Sub, suffix)
		}
		if prefix != nil {
			return prefix
		}

		// Some degenerate case like min > max or min < max < 0.
		// Handle as impossible match.
		return &Regexp{Op: OpNoMatch}
	}

	return re
}

// simplify1 implements Simplify for the unary OpStar,
// OpPlus, and OpQuest operators.  It returns the simple regexp
// equivalent to
//
//	Regexp{Op: op, Flags: flags, Sub: {sub}}
//
// under the assumption that sub is already simple, and
// without first allocating that structure.  If the regexp
// to be returned turns out to be equivalent to re, simplify1
// returns re instead.
//
// simplify1 is factored out of Simplify because the implementation
// for other operators generates these unary expressions.
// Letting them call simplify1 makes sure the expressions they
// generate are simple.
func simplify1(op Op, flags Flags, sub, re *Regexp) *Regexp {
	// Special case: repeat the empty string as much as
	// you want, but it's still the empty string.
	if sub.Op == OpEmptyMatch {
		return sub
	}
	// The operators are idempotent if the flags match.
	if op == sub.Op && flags&NonGreedy == sub.Flags&NonGreedy {
		return sub
	}
	if re != nil && re.Op == op && re.Flags&NonGreedy == flags&NonGreedy && sub == re.Sub[0] {
		return re
	}

	re = &Regexp{Op: op, Flags: flags}
	re.Sub = append(re.Sub0[:0], sub)
	return re
}
