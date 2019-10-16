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
	"errors"
	"fmt"
	"strings"
	"time"

	"internal/lazyregexp"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var pseudoVersionRE = lazyregexp.New(`^v[0-9]+\.(0\.0-|\d+\.\d+-([^+]*\.)?0\.)\d{14}-[A-Za-z0-9]+(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$`)

// PseudoVersion returns a pseudo-version for the given major version ("v1")
// preexisting older tagged version ("" or "v1.2.3" or "v1.2.3-pre"), revision time,
// and revision identifier (usually a 12-byte commit hash prefix).
func PseudoVersion(major, older string, t time.Time, rev string) string {
	if major == "" {
		major = "v0"
	}
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
	i := strings.LastIndex(older, ".") + 1
	v, patch := older[:i], older[i:]

	// Reassemble.
	return v + incDecimal(patch) + "-0." + segment + build
}

// incDecimal returns the decimal string incremented by 1.
func incDecimal(decimal string) string {
	// Scan right to left turning 9s to 0s until you find a digit to increment.
	digits := []byte(decimal)
	i := len(digits) - 1
	for ; i >= 0 && digits[i] == '9'; i-- {
		digits[i] = '0'
	}
	if i >= 0 {
		digits[i]++
	} else {
		// digits is all zeros
		digits[0] = '1'
		digits = append(digits, '0')
	}
	return string(digits)
}

// decDecimal returns the decimal string decremented by 1, or the empty string
// if the decimal is all zeroes.
func decDecimal(decimal string) string {
	// Scan right to left turning 0s to 9s until you find a digit to decrement.
	digits := []byte(decimal)
	i := len(digits) - 1
	for ; i >= 0 && digits[i] == '0'; i-- {
		digits[i] = '9'
	}
	if i < 0 {
		// decimal is all zeros
		return ""
	}
	if i == 0 && digits[i] == '1' && len(digits) > 1 {
		digits = digits[1:]
	} else {
		digits[i]--
	}
	return string(digits)
}

// IsPseudoVersion reports whether v is a pseudo-version.
func IsPseudoVersion(v string) bool {
	return strings.Count(v, "-") >= 2 && semver.IsValid(v) && pseudoVersionRE.MatchString(v)
}

// PseudoVersionTime returns the time stamp of the pseudo-version v.
// It returns an error if v is not a pseudo-version or if the time stamp
// embedded in the pseudo-version is not a valid time.
func PseudoVersionTime(v string) (time.Time, error) {
	_, timestamp, _, _, err := parsePseudoVersion(v)
	if err != nil {
		return time.Time{}, err
	}
	t, err := time.Parse("20060102150405", timestamp)
	if err != nil {
		return time.Time{}, &module.InvalidVersionError{
			Version: v,
			Pseudo:  true,
			Err:     fmt.Errorf("malformed time %q", timestamp),
		}
	}
	return t, nil
}

// PseudoVersionRev returns the revision identifier of the pseudo-version v.
// It returns an error if v is not a pseudo-version.
func PseudoVersionRev(v string) (rev string, err error) {
	_, _, rev, _, err = parsePseudoVersion(v)
	return
}

// PseudoVersionBase returns the canonical parent version, if any, upon which
// the pseudo-version v is based.
//
// If v has no parent version (that is, if it is "vX.0.0-[…]"),
// PseudoVersionBase returns the empty string and a nil error.
func PseudoVersionBase(v string) (string, error) {
	base, _, _, build, err := parsePseudoVersion(v)
	if err != nil {
		return "", err
	}

	switch pre := semver.Prerelease(base); pre {
	case "":
		// vX.0.0-yyyymmddhhmmss-abcdef123456 → ""
		if build != "" {
			// Pseudo-versions of the form vX.0.0-yyyymmddhhmmss-abcdef123456+incompatible
			// are nonsensical: the "vX.0.0-" prefix implies that there is no parent tag,
			// but the "+incompatible" suffix implies that the major version of
			// the parent tag is not compatible with the module's import path.
			//
			// There are a few such entries in the index generated by proxy.golang.org,
			// but we believe those entries were generated by the proxy itself.
			return "", &module.InvalidVersionError{
				Version: v,
				Pseudo:  true,
				Err:     fmt.Errorf("lacks base version, but has build metadata %q", build),
			}
		}
		return "", nil

	case "-0":
		// vX.Y.(Z+1)-0.yyyymmddhhmmss-abcdef123456 → vX.Y.Z
		// vX.Y.(Z+1)-0.yyyymmddhhmmss-abcdef123456+incompatible → vX.Y.Z+incompatible
		base = strings.TrimSuffix(base, pre)
		i := strings.LastIndexByte(base, '.')
		if i < 0 {
			panic("base from parsePseudoVersion missing patch number: " + base)
		}
		patch := decDecimal(base[i+1:])
		if patch == "" {
			// vX.0.0-0 is invalid, but has been observed in the wild in the index
			// generated by requests to proxy.golang.org.
			//
			// NOTE(bcmills): I cannot find a historical bug that accounts for
			// pseudo-versions of this form, nor have I seen such versions in any
			// actual go.mod files. If we find actual examples of this form and a
			// reasonable theory of how they came into existence, it seems fine to
			// treat them as equivalent to vX.0.0 (especially since the invalid
			// pseudo-versions have lower precedence than the real ones). For now, we
			// reject them.
			return "", &module.InvalidVersionError{
				Version: v,
				Pseudo:  true,
				Err:     fmt.Errorf("version before %s would have negative patch number", base),
			}
		}
		return base[:i+1] + patch + build, nil

	default:
		// vX.Y.Z-pre.0.yyyymmddhhmmss-abcdef123456 → vX.Y.Z-pre
		// vX.Y.Z-pre.0.yyyymmddhhmmss-abcdef123456+incompatible → vX.Y.Z-pre+incompatible
		if !strings.HasSuffix(base, ".0") {
			panic(`base from parsePseudoVersion missing ".0" before date: ` + base)
		}
		return strings.TrimSuffix(base, ".0") + build, nil
	}
}

var errPseudoSyntax = errors.New("syntax error")

func parsePseudoVersion(v string) (base, timestamp, rev, build string, err error) {
	if !IsPseudoVersion(v) {
		return "", "", "", "", &module.InvalidVersionError{
			Version: v,
			Pseudo:  true,
			Err:     errPseudoSyntax,
		}
	}
	build = semver.Build(v)
	v = strings.TrimSuffix(v, build)
	j := strings.LastIndex(v, "-")
	v, rev = v[:j], v[j+1:]
	i := strings.LastIndex(v, "-")
	if j := strings.LastIndex(v, "."); j > i {
		base = v[:j] // "vX.Y.Z-pre.0" or "vX.Y.(Z+1)-0"
		timestamp = v[j+1:]
	} else {
		base = v[:i] // "vX.0.0"
		timestamp = v[i+1:]
	}
	return base, timestamp, rev, build, nil
}
