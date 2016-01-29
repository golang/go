// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt"
	"os"
	"strconv"
	"sync"
)

// matcher sanitizes, uniques, and filters names of subtests and subbenchmarks.
type matcher struct {
	filter    string
	matchFunc func(pat, str string) (bool, error)

	mu       sync.Mutex
	subNames map[string]int64
}

// TODO: fix test_main to avoid race and improve caching.
var matchMutex sync.Mutex

func newMatcher(matchString func(pat, str string) (bool, error), pattern, name string) *matcher {
	// Verify filters before doing any processing.
	if _, err := matchString(pattern, "non-empty"); err != nil {
		fmt.Fprintf(os.Stderr, "testing: invalid regexp for %s: %s\n", name, err)
		os.Exit(1)
	}
	return &matcher{
		filter:    pattern,
		matchFunc: matchString,
		subNames:  map[string]int64{},
	}
}

func (m *matcher) fullName(c *common, subname string) (name string, ok bool) {
	name = subname

	m.mu.Lock()
	defer m.mu.Unlock()

	if c != nil && c.level > 0 {
		name = m.unique(c.name, rewrite(subname))
	}

	matchMutex.Lock()
	defer matchMutex.Unlock()

	if c != nil && c.level == 0 {
		if matched, _ := m.matchFunc(m.filter, subname); !matched {
			return name, false
		}
	}
	return name, true
}

// unique creates a unique name for the given parent and subname by affixing it
// with one ore more counts, if necessary.
func (m *matcher) unique(parent, subname string) string {
	name := fmt.Sprintf("%s/%s", parent, subname)
	empty := subname == ""
	for {
		next, exists := m.subNames[name]
		if !empty && !exists {
			m.subNames[name] = 1 // next count is 1
			return name
		}
		// Name was already used. We increment with the count and append a
		// string with the count.
		m.subNames[name] = next + 1

		// Add a count to guarantee uniqueness.
		name = fmt.Sprintf("%s#%02d", name, next)
		empty = false
	}
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
