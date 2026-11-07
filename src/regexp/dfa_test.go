// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regexp

import (
	"math/rand/v2"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
	"unsafe"
)

var dfaPatterns = []string{
	`a`,
	`abc`,
	`a*`,
	`a+?`,
	`()`,
	`x*$`,
	`^`,
	`$`,
	`^abc`,
	`abc$`,
	`\Aab`,
	`cd\z`,
	`(?m)^cd`,
	`(?m)ab$`,
	`(?m)^$`,
	`\bab\b`,
	`\Bb\B`,
	`\b`,
	`(a|ab)(c|bcd)(d*)`,
	`(a+|b+)*c`,
	`a.*?b`,
	`a.*b`,
	`(?s)a.b`,
	`(?i)ÉLLO`,
	`\p{Greek}+`,
	`[^a-z]+`,
	`é|abc|β`,
	`ab|cd|`,
	`latency=(\d+)`,
	`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`,
	`\d{1,3}\.\d{1,3}\b`,
}

var dfaInputs = []string{
	"",
	"a",
	"abcd",
	"xabcdx",
	"ab\ncd\n",
	"\ncd\nab",
	"aab abb ab",
	"héllo wörld ÉLLO",
	"αβγ abc δ",
	"\xff\xfeab\xc3",
	"\xe2\x82abc\xe2\x82\xac",
	"user@example.com latency=42 10.0.0.1 x",
	"aaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
}

// nfaIndex runs the engine find used before the DFA, as the reference for it.
func nfaIndex(re *Regexp, s string, pos int) []int {
	if re.onepass != nil {
		return re.doOnePass(nil, nil, s, pos, 2, nil)
	}
	m := re.get()
	defer re.put(m)
	i, _ := m.inputs.init(nil, nil, s)
	m.init(2)
	if !m.match(i, pos) {
		return nil
	}
	return slices.Clone(m.matchcap[:2])
}

func dfaIndex(t *testing.T, re *Regexp, b []byte, s string, pos int, earliest bool) []int {
	t.Helper()
	start, end, ok := re.dfaBounds(b, s, pos, earliest)
	if !ok {
		t.Fatalf("%#q: DFA gave up on %q at %d", re, s, pos)
	}
	if end < 0 {
		return nil
	}
	return []int{start, end}
}

func TestDFAMatchesNFA(t *testing.T) {
	for _, pat := range dfaPatterns {
		re := MustCompile(pat)
		for _, in := range dfaInputs {
			for pos := 0; pos <= len(in); pos++ {
				want := nfaIndex(re, in, pos)
				if got := dfaIndex(t, re, nil, in, pos, false); !slices.Equal(got, want) {
					t.Errorf("%#q on %q at %d: string DFA = %v, NFA = %v", pat, in, pos, got, want)
				}
				if got := dfaIndex(t, re, []byte(in), "", pos, false); !slices.Equal(got, want) {
					t.Errorf("%#q on %q at %d: bytes DFA = %v, NFA = %v", pat, in, pos, got, want)
				}
				if got := dfaIndex(t, re, nil, in, pos, true); (got == nil) != (want == nil) {
					t.Errorf("%#q on %q at %d: earliest DFA = %v, NFA = %v", pat, in, pos, got, want)
				}
			}
		}
	}
}

func TestDFAFindAllMatchesNFA(t *testing.T) {
	in := strings.Repeat("ab\ncd ÉLLO αβγ user@example.com latency=42 ", 100)
	for _, pat := range dfaPatterns {
		re := MustCompile(pat)
		var want [][]int
		for pos, prevEnd := 0, -1; pos <= len(in); {
			m := nfaIndex(re, in, pos)
			if m == nil {
				break
			}
			if m[1] == pos {
				if m[0] != prevEnd {
					want = append(want, m)
				}
				if _, w := (&inputString{str: in}).step(pos); w > 0 {
					pos += w
				} else {
					pos = len(in) + 1
				}
			} else {
				want = append(want, m)
				pos = m[1]
			}
			prevEnd = m[1]
		}
		if got := re.FindAllStringIndex(in, -1); !slices.EqualFunc(got, want, slices.Equal) {
			t.Errorf("%#q: FindAllStringIndex = %d matches, NFA = %d matches", pat, len(got), len(want))
		}
	}
}

func TestDFASubmatchesAfterDFA(t *testing.T) {
	re := MustCompile(`latency=(\d+)ms`)
	in := strings.Repeat("x", re.maxBitStateLen) + "latency=42ms"
	if !re.useDFA(len(in), 4) {
		t.Fatalf("useDFA(%d, 4) = false, want true", len(in))
	}
	want := []string{"latency=42ms", "42"}
	if got := re.FindStringSubmatch(in); !slices.Equal(got, want) {
		t.Errorf("FindStringSubmatch = %q, want %q", got, want)
	}
	if got := re.FindStringSubmatch(in[:len(in)-2]); got != nil {
		t.Errorf("FindStringSubmatch without the suffix = %q, want nil", got)
	}
}

func TestDFAGivesUpWhenCacheIsFull(t *testing.T) {
	// This pattern needs about 2^13 DFA states, more than dfaMaxStates.
	re := MustCompile(`(a|b)*a(a|b){12}`)
	r := rand.New(rand.NewPCG(1, 2))
	var sb strings.Builder
	for range 1 << 16 {
		sb.WriteByte("ab"[r.IntN(2)])
	}
	in := sb.String() + "c"
	if _, _, ok := re.dfaBounds(nil, in, 0, false); ok {
		t.Fatal("dfaBounds did not give up, want a full state cache")
	}
	if got := re.dfaProg().skip.Load(); got != dfaBackoff {
		t.Errorf("searches skipped after giving up = %d, want %d", got, dfaBackoff)
	}
	if got, want := re.FindStringIndex(in), nfaIndex(re, in, 0); !slices.Equal(got, want) {
		t.Errorf("FindStringIndex after the DFA gave up = %v, want %v", got, want)
	}
	if !re.MatchString(in) {
		t.Error("MatchString after the DFA gave up = false, want true")
	}
}

func TestDFANotUsed(t *testing.T) {
	for _, tc := range []struct {
		re      *Regexp
		n, ncap int
	}{
		{MustCompilePOSIX(`a+`), 1 << 20, 2},
		{MustCompile(`abc`), 1 << 20, 2},
		{MustCompile(`^abc(d|e)`), 1 << 20, 2},
		{MustCompile(`abc(d|e)`), dfaMinPrefixInput - 1, 2},
		{MustCompile(`a+(b)`), 10, 4},
	} {
		if tc.re.useDFA(tc.n, tc.ncap) {
			t.Errorf("%#q: useDFA(%d, %d) = true, want false", tc.re, tc.n, tc.ncap)
		}
	}
}

func TestDFAProgIsReleased(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("cleanups may not run before the deadline on wasm")
	}
	re := MustCompile(`[a-c]+x`)
	re.MatchString(strings.Repeat("abc", 100))
	k := uintptr(unsafe.Pointer(re.prog))
	if _, ok := dfaProgs.Load(k); !ok {
		t.Fatal("no DFA entry after a search")
	}
	re = nil
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); {
		runtime.GC()
		if _, ok := dfaProgs.Load(k); !ok {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Error("DFA entry still present after the regexp was collected")
}

func TestDFABackoffExpires(t *testing.T) {
	re := MustCompile(`a+b`)
	if _, end, ok := re.dfaBounds(nil, "xaab", 0, false); !ok || end != 4 {
		t.Fatalf("dfaBounds = end %d, ok %v, want 4, true", end, ok)
	}
	if got := re.dfaProg().skip.Load(); got != 0 {
		t.Fatalf("searches skipped after a successful search = %d, want 0", got)
	}
	re.dfaProg().skip.Store(1)
	if _, _, ok := re.dfaBounds(nil, "xaab", 0, false); ok {
		t.Error("dfaBounds ran during the backoff, want it skipped")
	}
	if got, want := re.FindStringIndex("xaab"), []int{1, 4}; !slices.Equal(got, want) {
		t.Errorf("FindStringIndex during the backoff = %v, want %v", got, want)
	}
	start, end, ok := re.dfaBounds(nil, "xaab", 0, false)
	if !ok || start != 1 || end != 4 {
		t.Errorf("dfaBounds after the backoff = [%d %d], ok %v, want [1 4], true", start, end, ok)
	}
}
