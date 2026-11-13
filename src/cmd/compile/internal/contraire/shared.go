// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package contraire

// JSON encoding of messages emitted from both workers and the driver.
type output struct {
	TrialsDone  int         // The number of trials completed.
	Next        *scoredProg // A write-ahead of the program about to be type-checked.
	ParseErrors int         // The number of generated programs with parse errors.
	TypeErrors  int         // The number of generated programs with type errors.
	Successes   int         // The number of generated programs without errors.
	Panic       *scoredProg // The lowest-score program with a panic failure.
	Crash       *scoredProg // The lowest-score program with a crash failure.
	Hang        *scoredProg // The lowest-score program with a hang failure.
}

type scoredProg struct {
	Source string
	Score  int
}

func (a *output) merge(b *output) {
	a.TrialsDone += b.TrialsDone
	if a.Next != nil || b.Next != nil {
		panic("unexpected Next field")
	}
	a.ParseErrors += b.ParseErrors
	a.TypeErrors += b.TypeErrors
	a.Successes += b.Successes
	a.Panic = min(a.Panic, b.Panic)
	a.Crash = min(a.Crash, b.Crash)
	a.Hang = min(a.Hang, b.Hang)
}

func min(a *scoredProg, b *scoredProg) *scoredProg {
	switch {
	case a == nil:
		return b
	case b == nil:
		return a
	case a.Score <= b.Score:
		return a
	default:
		return b
	}
}
