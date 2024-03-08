// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

// matcher sanitizes, uniques, and filters names of subtests and subbenchmarks.
type matcher struct {
	filter    filterMatch
	skip      filterMatch
	matchFunc func(pat, str string) (bool, error)

	mu sync.Mutex

	// subNames is used to deduplicate subtest names.
	// Each key is the subtest name joined to the deduplicated name of the parent test.
	// Each value is the count of the number of occurrences of the given subtest name
	// already seen.
	subNames map[string]int32
}

type filterMatch interface {
	// matches checks the name against the receiver's pattern strings using the
	// given match function.
	matches(name []string, matchString func(pat, str string) (bool, error)) (ok, partial bool)

	// verify checks that the receiver's pattern strings are valid filters by
	// calling the given match function.
	verify(name string, matchString func(pat, str string) (bool, error)) error
}

// simpleMatch matches a test name if all of the pattern strings match in
// sequence.
type simpleMatch []string

// alternationMatch matches a test name if one of the alternations match.
type alternationMatch []filterMatch

func allMatcher() *matcher {
	return newMatcher(nil, "", "", "")
}

func newMatcher(matchString func(pat, str string) (bool, error), patterns, name, skips string) *matcher {
	var filter, skip filterMatch
	if patterns == "" {
		filter = simpleMatch{} // always partial true
	} else {
		filter = splitRegexp(patterns)
		if err := filter.verify(name, matchString); err != nil {
			fmt.Fprintf(os.Stderr, "testing: invalid regexp for %s\n", err)
			os.Exit(1)
		}
	}
	if skips == "" {
		skip = alternationMatch{} // always false
	} else {
		skip = splitRegexp(skips)
		if err := skip.verify("-test.skip", matchString); err != nil {
			fmt.Fprintf(os.Stderr, "testing: invalid regexp for %v\n", err)
			os.Exit(1)
		}
	}
	return &matcher{
		filter:    filter,
		skip:      skip,
		matchFunc: matchString,
		subNames:  map[string]int32{},
	}
}

func (m *matcher) fullName(c *common, subname string) (name string, ok, partial bool) {
	name = subname

	m.mu.Lock()
	defer m.mu.Unlock()

	if c != nil && c.level > 0 {
		name = m.unique(c.name, rewrite(subname))
	}

	// We check the full array of paths each time to allow for the case that a pattern contains a '/'.
	elem := strings.Split(name, "/")

	// filter must match.
	// accept partial match that may produce full match later.
	ok, partial = m.filter.matches(elem, m.matchFunc)
	if !ok {
		return name, false, false
	}

	// skip must not match.
	// ignore partial match so we can get to more precise match later.
	skip, partialSkip := m.skip.matches(elem, m.matchFunc)
	if skip && !partialSkip {
		return name, false, false
	}

	return name, ok, partial
}

// clearSubNames clears the matcher's internal state, potentially freeing
// memory. After this is called, T.Name may return the same strings as it did
// for earlier subtests.
func (m *matcher) clearSubNames() {
	m.mu.Lock()
	defer m.mu.Unlock()
	clear(m.subNames)
}

func (m simpleMatch) matches(name []string, matchString func(pat, str string) (bool, error)) (ok, partial bool) {
	for i, s := range name {
		if i >= len(m) {
			break
		}
		if ok, _ := matchString(m[i], s); !ok {
			return false, false
		}
	}
	return true, len(name) < len(m)
}

func (m simpleMatch) verify(name string, matchString func(pat, str string) (bool, error)) error {
	for i, s := range m {
		m[i] = rewrite(s)
	}
	// Verify filters before doing any processing.
	for i, s := range m {
		if _, err := matchString(s, "non-empty"); err != nil {
			return fmt.Errorf("element %d of %s (%q): %s", i, name, s, err)
		}
	}
	return nil
}

func (m alternationMatch) matches(name []string, matchString func(pat, str string) (bool, error)) (ok, partial bool) {
	for _, m := range m {
		if ok, partial = m.matches(name, matchString); ok {
			return ok, partial
		}
	}
	return false, false
}

func (m alternationMatch) verify(name string, matchString func(pat, str string) (bool, error)) error {
	for i, m := range m {
		if err := m.verify(name, matchString); err != nil {
			return fmt.Errorf("alternation %d of %s", i, err)
		}
	}
	return nil
}

func splitRegexp(s string) filterMatch {
	a := make(simpleMatch, 0, strings.Count(s, "/"))
	b := make(alternationMatch, 0, strings.Count(s, "|"))
	cs := 0
	cp := 0
	for i := 0; i < len(s); {
		switch s[i] {
		case '[':
			cs++
		case ']':
			if cs--; cs < 0 { // An unmatched ']' is legal.
				cs = 0
			}
		case '(':
			if cs == 0 {
				cp++
			}
		case ')':
			if cs == 0 {
				cp--
			}
		case '\\':
			i++
		case '/':
			if cs == 0 && cp == 0 {
				a = append(a, s[:i])
				s = s[i+1:]
				i = 0
				continue
			}
		case '|':
			if cs == 0 && cp == 0 {
				a = append(a, s[:i])
				s = s[i+1:]
				i = 0
				b = append(b, a)
				a = make(simpleMatch, 0, len(a))
				continue
			}
		}
		i++
	}

	a = append(a, s)
	if len(b) == 0 {
		return a
	}
	return append(b, a)
}

// unique creates a unique name for the given parent and subname by affixing it
// with one or more counts, if necessary.
func (m *matcher) unique(parent, subname string) string {
	base := parent + "/" + subname

	for {
		n := m.subNames[base]
		if n < 0 {
			panic("subtest count overflow")
		}
		m.subNames[base] = n + 1

		if n == 0 && subname != "" {
			prefix, nn := parseSubtestNumber(base)
			if len(prefix) < len(base) && nn < m.subNames[prefix] {
				// This test is explicitly named like "parent/subname#NN",
				// and #NN was already used for the NNth occurrence of "parent/subname".
				// Loop to add a disambiguating suffix.
				continue
			}
			return base
		}

		name := fmt.Sprintf("%s#%02d", base, n)
		if m.subNames[name] != 0 {
			// This is the nth occurrence of base, but the name "parent/subname#NN"
			// collides with the first occurrence of a subtest *explicitly* named
			// "parent/subname#NN". Try the next number.
			continue
		}

		return name
	}
}

// parseSubtestNumber splits a subtest name into a "#%02d"-formatted int32
// suffix (if present), and a prefix preceding that suffix (always).
func parseSubtestNumber(s string) (prefix string, nn int32) {
	i := strings.LastIndex(s, "#")
	if i < 0 {
		return s, 0
	}

	prefix, suffix := s[:i], s[i+1:]
	if len(suffix) < 2 || (len(suffix) > 2 && suffix[0] == '0') {
		// Even if suffix is numeric, it is not a possible output of a "%02" format
		// string: it has either too few digits or too many leading zeroes.
		return s, 0
	}
	if suffix == "00" {
		if !strings.HasSuffix(prefix, "/") {
			// We only use "#00" as a suffix for subtests named with the empty
			// string — it isn't a valid suffix if the subtest name is non-empty.
			return s, 0
		}
	}

	n, err := strconv.ParseInt(suffix, 10, 32)
	if err != nil || n < 0 {
		return s, 0
	}
	return prefix, int32(n)
}

// rewrite rewrites a subname to having only printable characters and no white
// space.
func rewrite(s string) string {
	b := []byte{}
	for _, r := range s {
		switch {
		case isSpace(r):
			b = append(b, '_')
		case !strconv.IsPrint(r):
			s := strconv.QuoteRune(r)
			b = append(b, s[1:len(s)-1]...)
		default:
			b = append(b, string(r)...)
		}
	}
	return string(b)
}

func isSpace(r rune) bool {
	if r < 0x2000 {
		switch r {
		// Note: not the same as Unicode Z class.
		case '\t', '\n', '\v', '\f', '\r', ' ', 0x85, 0xA0, 0x1680:
			return true
		}
	} else {
		if r <= 0x200a {
			return true
		}
		switch r {
		case 0x2028, 0x2029, 0x202f, 0x205f, 0x3000:
			return true
		}
	}
	return false
}
