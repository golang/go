// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ((darwin || dragonfly || freebsd || (js && wasm) || wasip1 || (!android && linux) || netbsd || openbsd || solaris) && ((!cgo && !darwin) || osusergo)) || aix || illumos

package user

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
)

func listGroupsFromReader(u *User, r io.Reader) ([]string, error) {
	if u.Username == "" {
		return nil, errors.New("user: list groups: empty username")
	}
	primaryGid, err := strconv.Atoi(u.Gid)
	if err != nil {
		return nil, fmt.Errorf("user: list groups for %s: invalid gid %q", u.Username, u.Gid)
	}

	userCommas := []byte("," + u.Username + ",")  // ,john,
	userFirst := userCommas[1:]                   // john,
	userLast := userCommas[:len(userCommas)-1]    // ,john
	userOnly := userCommas[1 : len(userCommas)-1] // john

	// Add primary Gid first.
	groups := []string{u.Gid}

	rd := bufio.NewReader(r)
	done := false
	for !done {
		line, err := rd.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				done = true
			} else {
				return groups, err
			}
		}

		// Look for username in the list of users. If user is found,
		// append the GID to the groups slice.

		// There's no spec for /etc/passwd or /etc/group, but we try to follow
		// the same rules as the glibc parser, which allows comments and blank
		// space at the beginning of a line.
		line = bytes.TrimSpace(line)
		if len(line) == 0 || line[0] == '#' ||
			// If you search for a gid in a row where the group
			// name (the first field) starts with "+" or "-",
			// glibc fails to find the record, and so should we.
			line[0] == '+' || line[0] == '-' {
			continue
		}

		// Format of /etc/group is
		// 	groupname:password:GID:user_list
		// for example
		// 	wheel:x:10:john,paul,jack
		//	tcpdump:x:72:
		listIdx := bytes.LastIndexByte(line, ':')
		if listIdx == -1 || listIdx == len(line)-1 {
			// No commas, or empty group list.
			continue
		}
		if bytes.Count(line[:listIdx], colon) != 2 {
			// Incorrect number of colons.
			continue
		}
		list := line[listIdx+1:]
		// Check the list for user without splitting or copying.
		if !(bytes.Equal(list, userOnly) || bytes.HasPrefix(list, userFirst) || bytes.HasSuffix(list, userLast) || bytes.Contains(list, userCommas)) {
			continue
		}

		// groupname:password:GID
		parts := bytes.Split(line[:listIdx], colon)
		if len(parts) != 3 || len(parts[0]) == 0 {
			continue
		}
		gid := string(parts[2])
		// Make sure it's numeric and not the same as primary GID.
		numGid, err := strconv.Atoi(gid)
		if err != nil || numGid == primaryGid {
			continue
		}

		groups = append(groups, gid)
	}

	return groups, nil
}

func listGroups(u *User) ([]string, error) {
	f, err := os.Open(groupFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return listGroupsFromReader(u, f)
}
