// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testtrace

import (
	"bufio"
	"bytes"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Expectation represents the expected result of some operation.
type Expectation struct {
	failure      bool
	errorMatcher *regexp.Regexp
}

// ExpectSuccess returns an Expectation that trivially expects success.
func ExpectSuccess() *Expectation {
	return new(Expectation)
}

// Check validates whether err conforms to the expectation. Returns
// an error if it does not conform.
//
// Conformance means that if failure is true, then err must be non-nil.
// If err is non-nil, then it must match errorMatcher.
func (e *Expectation) Check(err error) error {
	if !e.failure && err != nil {
		return fmt.Errorf("unexpected error while reading the trace: %v", err)
	}
	if e.failure && err == nil {
		return fmt.Errorf("expected error while reading the trace: want something matching %q, got none", e.errorMatcher)
	}
	if e.failure && err != nil && !e.errorMatcher.MatchString(err.Error()) {
		return fmt.Errorf("unexpected error while reading the trace: want something matching %q, got %s", e.errorMatcher, err.Error())
	}
	return nil
}

// ParseExpectation parses the serialized form of an Expectation.
func ParseExpectation(data []byte) (*Expectation, error) {
	exp := new(Expectation)
	s := bufio.NewScanner(bytes.NewReader(data))
	if s.Scan() {
		c := strings.SplitN(s.Text(), " ", 2)
		switch c[0] {
		case "SUCCESS":
		case "FAILURE":
			exp.failure = true
			if len(c) != 2 {
				return exp, fmt.Errorf("bad header line for FAILURE: %q", s.Text())
			}
			matcher, err := parseMatcher(c[1])
			if err != nil {
				return exp, err
			}
			exp.errorMatcher = matcher
		default:
			return exp, fmt.Errorf("bad header line: %q", s.Text())
		}
		return exp, nil
	}
	return exp, s.Err()
}

func parseMatcher(quoted string) (*regexp.Regexp, error) {
	pattern, err := strconv.Unquote(quoted)
	if err != nil {
		return nil, fmt.Errorf("malformed pattern: not correctly quoted: %s: %v", quoted, err)
	}
	matcher, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("malformed pattern: not a valid regexp: %s: %v", pattern, err)
	}
	return matcher, nil
}
