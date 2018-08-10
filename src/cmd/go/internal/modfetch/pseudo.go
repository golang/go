// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Pseudo-versions
//
// Code authors are expected to tag the revisions they want users to use,
// including prereleases. However, not all authors tag versions at all,
// and not all commits a user might want to try will have tags.
// A pseudo-version is a version with a special form that allows us to
// address an untagged commit and order that version with respect to
// other versions we might encounter.
//
// A pseudo-version takes one of the general forms:
//
//	(1) vX.0.0-yyyymmddhhmmss-abcdef123456
//	(2) vX.Y.(Z+1)-0.yyyymmddhhmmss-abcdef123456
//	(3) vX.Y.(Z+1)-0.yyyymmddhhmmss-abcdef123456+incompatible
//	(4) vX.Y.Z-pre.0.yyyymmddhhmmss-abcdef123456
//	(5) vX.Y.Z-pre.0.yyyymmddhhmmss-abcdef123456+incompatible
//
// If there is no recently tagged version with the right major version vX,
// then form (1) is used, creating a space of pseudo-versions at the bottom
// of the vX version range, less than any tagged version, including the unlikely v0.0.0.
//
// If the most recent tagged version before the target commit is vX.Y.Z or vX.Y.Z+incompatible,
// then the pseudo-version uses form (2) or (3), making it a prerelease for the next
// possible semantic version after vX.Y.Z. The leading 0 segment in the prerelease string
// ensures that the pseudo-version compares less than possible future explicit prereleases
// like vX.Y.(Z+1)-rc1 or vX.Y.(Z+1)-1.
//
// If the most recent tagged version before the target commit is vX.Y.Z-pre or vX.Y.Z-pre+incompatible,
// then the pseudo-version uses form (4) or (5), making it a slightly later prerelease.

package modfetch

import (
	"cmd/go/internal/semver"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// PseudoVersion returns a pseudo-version for the given major version ("v1")
// preexisting older tagged version ("" or "v1.2.3" or "v1.2.3-pre"), revision time,
// and revision identifier (usually a 12-byte commit hash prefix).
func PseudoVersion(major, older string, t time.Time, rev string) string {
	if major == "" {
		major = "v0"
	}
	major = strings.TrimSuffix(major, "-unstable") // make gopkg.in/macaroon-bakery.v2-unstable use "v2"
	segment := fmt.Sprintf("%s-%s", t.UTC().Format("20060102150405"), rev)
	build := semver.Build(older)
	older = semver.Canonical(older)
	if older == "" {
		return major + ".0.0-" + segment // form (1)
	}
	if semver.Prerelease(older) != "" {
		return older + ".0." + segment + build // form (4), (5)
	}

	// Form (2), (3).
	// Extract patch from vMAJOR.MINOR.PATCH
	v := older[:len(older)]
	i := strings.LastIndex(v, ".") + 1
	v, patch := v[:i], v[i:]

	// Increment PATCH by adding 1 to decimal:
	// scan right to left turning 9s to 0s until you find a digit to increment.
	// (Number might exceed int64, but math/big is overkill.)
	digits := []byte(patch)
	for i = len(digits) - 1; i >= 0 && digits[i] == '9'; i-- {
		digits[i] = '0'
	}
	if i >= 0 {
		digits[i]++
	} else {
		// digits is all zeros
		digits[0] = '1'
		digits = append(digits, '0')
	}
	patch = string(digits)

	// Reassemble.
	return v + patch + "-0." + segment + build
}

var pseudoVersionRE = regexp.MustCompile(`^v[0-9]+\.(0\.0-|\d+\.\d+-([^+]*\.)?0\.)\d{14}-[A-Za-z0-9]+(\+incompatible)?$`)

// IsPseudoVersion reports whether v is a pseudo-version.
func IsPseudoVersion(v string) bool {
	return strings.Count(v, "-") >= 2 && semver.IsValid(v) && pseudoVersionRE.MatchString(v)
}

// PseudoVersionTime returns the time stamp of the pseudo-version v.
// It returns an error if v is not a pseudo-version or if the time stamp
// embedded in the pseudo-version is not a valid time.
func PseudoVersionTime(v string) (time.Time, error) {
	timestamp, _, err := parsePseudoVersion(v)
	t, err := time.Parse("20060102150405", timestamp)
	if err != nil {
		return time.Time{}, fmt.Errorf("pseudo-version with malformed time %s: %q", timestamp, v)
	}
	return t, nil
}

// PseudoVersionRev returns the revision identifier of the pseudo-version v.
// It returns an error if v is not a pseudo-version.
func PseudoVersionRev(v string) (rev string, err error) {
	_, rev, err = parsePseudoVersion(v)
	return
}

func parsePseudoVersion(v string) (timestamp, rev string, err error) {
	if !IsPseudoVersion(v) {
		return "", "", fmt.Errorf("malformed pseudo-version %q", v)
	}
	v = strings.TrimSuffix(v, "+incompatible")
	j := strings.LastIndex(v, "-")
	v, rev = v[:j], v[j+1:]
	i := strings.LastIndex(v, "-")
	if j := strings.LastIndex(v, "."); j > i {
		timestamp = v[j+1:]
	} else {
		timestamp = v[i+1:]
	}
	return timestamp, rev, nil
}
