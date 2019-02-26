// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"fmt"
	"os"
	pathpkg "path"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/get"
	"cmd/go/internal/module"
)

// notaryShouldVerify reports whether the notary should be used for path,
// given the GONOVERIFY setting.
func notaryShouldVerify(path, GONOVERIFY string) (bool, error) {
	if GONOVERIFY == "off" {
		return false, nil
	}
	for GONOVERIFY != "" {
		var pattern string
		i := strings.Index(GONOVERIFY, ",")
		if i < 0 {
			pattern, GONOVERIFY = GONOVERIFY, ""
		} else {
			pattern, GONOVERIFY = GONOVERIFY[:i], GONOVERIFY[i+1:]
		}
		if pattern == "" {
			continue
		}
		n := strings.Count(pattern, "/") + 1
		prefix := path
		for i := 0; i < len(prefix); i++ {
			if prefix[i] == '/' {
				n--
				if n == 0 {
					prefix = prefix[:i]
					break
				}
			}
		}
		if n > 1 {
			continue
		}
		matched, err := pathpkg.Match(pattern, prefix)
		if err != nil {
			// Note that path.Match does not guarantee to detect
			// pattern errors. It usually depends on whether the
			// given text (prefix in this case) matches enough of
			// the pattern to reach the error. So this will only
			// trigger on malformed patterns that are "close enough" to prefix.
			return false, fmt.Errorf("malformed GONOVERIFY pattern: %s", pattern)
		}
		if matched {
			return false, nil
		}
	}
	return true, nil
}

// useNotary reports whether to use the notary for the given module.
func useNotary(mod module.Version) bool {
	if get.Insecure {
		return false
	}
	wantNotary, err := notaryShouldVerify(mod.Path, os.Getenv("GONOVERIFY"))
	if err != nil {
		base.Fatalf("%v", err)
	}

	// TODO(rsc): return wantNotary. See #30601.
	//
	// This code must be deleted when goSumPin is deleted.
	// goSumPin is only a partial notary simulation, so we don't return true from
	// useNotary when we don't have an entry for that module.
	// This differs from the real notary, which will be authoritative
	// for everything it is asked for. When goSumPin is removed,
	// this function body should end here with "return wantNotary".

	_ = goSumPin // read TODO above if goSumPin is gone
	return wantNotary && notaryHashes(mod) != nil
}

// notaryHashes fetches hashes for mod from the notary.
// The caller must have checked that useNotary(mod) is true.
func notaryHashes(mod module.Version) []string {
	// For testing, hard-code this result.
	if mod.Path == "rsc.io/badsum" {
		switch mod.Version {
		case "v1.0.0":
			return []string{"h1:6/o+QJfe6mFSNuegDihphabcvR94anXQk/qq7Enr19U="}
		case "v1.0.0/go.mod":
			return []string{"h1:avOsLUJaHavllihBU9qCTW37z64ypkZjqZg8O16JLVY="}
		case "v1.0.1":
			return []string{"h1:S7G9Ikksx7htnFivDrUOv8xI0kIdAf15gLt97Gy//Zk="}
		case "v1.0.1/go.mod":
			return []string{"h1:avOsLUJaHavllihBU9qCTW37z64ypkZjqZg8O16JLVY="}
		}
	}

	// Until the notary is ready, simulate contacting the notary by
	// looking in the known hash list goSumPin in pin.go.
	// Entries not listed in goSumPin are treated as "not for the notary",
	// but once the real notary is added, they should be treated as
	// "failed to verify".
	//
	// TODO(rsc): Once the notary is ready, this function should be
	// rewritten to use it. See #30601.
	i := strings.Index(goSumPin, "\n"+mod.Path+"\n")
	if i < 0 {
		return nil
	}
	wantGoSum := false
	if strings.HasSuffix(mod.Version, "/go.mod") {
		wantGoSum = true
		mod.Version = strings.TrimSuffix(mod.Version, "/go.mod")
	}
	versions := goSumPin[i+1+len(mod.Path)+1:]
	var lastSum, lastGoSum string
	for {
		i := strings.Index(versions, "\n")
		if i < 0 {
			break
		}
		line := versions[:i]
		versions = versions[i+1:]
		if !strings.HasPrefix(line, " ") {
			break
		}
		f := strings.Fields(line)
		if len(f) < 3 {
			break
		}
		if f[1] == "-" {
			f[1] = lastSum
		} else {
			lastSum = f[1]
		}
		if f[2] == "-" {
			f[2] = lastGoSum
		} else {
			lastGoSum = f[2]
		}
		if f[0] == mod.Version {
			if wantGoSum {
				return []string{f[2]}
			}
			return []string{f[1]}
		}
	}
	return nil
}
