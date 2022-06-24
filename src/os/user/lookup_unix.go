// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ((unix && !android) || (js && wasm)) && (!cgo || osusergo)

package user

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"os"
	"strconv"
	"strings"
)

const userFile = "/etc/passwd"

// lineFunc returns a value, an error, or (nil, nil) to skip the row.
type lineFunc func(line []byte) (v any, err error)

// readColonFile parses r as an /etc/group or /etc/passwd style file, running
// fn for each row. readColonFile returns a value, an error, or (nil, nil) if
// the end of the file is reached without a match.
//
// readCols is the minimum number of colon-separated fields that will be passed
// to fn; in a long line additional fields may be silently discarded.
func readColonFile(r io.Reader, fn lineFunc, readCols int) (v any, err error) {
	rd := bufio.NewReader(r)

	// Read the file line-by-line.
	for {
		var isPrefix bool
		var wholeLine []byte

		// Read the next line. We do so in chunks (as much as reader's
		// buffer is able to keep), check if we read enough columns
		// already on each step and store final result in wholeLine.
		for {
			var line []byte
			line, isPrefix, err = rd.ReadLine()

			if err != nil {
				// We should return (nil, nil) if EOF is reached
				// without a match.
				if err == io.EOF {
					err = nil
				}
				return nil, err
			}

			// Simple common case: line is short enough to fit in a
			// single reader's buffer.
			if !isPrefix && len(wholeLine) == 0 {
				wholeLine = line
				break
			}

			wholeLine = append(wholeLine, line...)

			// Check if we read the whole line (or enough columns)
			// already.
			if !isPrefix || bytes.Count(wholeLine, []byte{':'}) >= readCols {
				break
			}
		}

		// There's no spec for /etc/passwd or /etc/group, but we try to follow
		// the same rules as the glibc parser, which allows comments and blank
		// space at the beginning of a line.
		wholeLine = bytes.TrimSpace(wholeLine)
		if len(wholeLine) == 0 || wholeLine[0] == '#' {
			continue
		}
		v, err = fn(wholeLine)
		if v != nil || err != nil {
			return
		}

		// If necessary, skip the rest of the line
		for ; isPrefix; _, isPrefix, err = rd.ReadLine() {
			if err != nil {
				// We should return (nil, nil) if EOF is reached without a match.
				if err == io.EOF {
					err = nil
				}
				return nil, err
			}
		}
	}
}

func matchGroupIndexValue(value string, idx int) lineFunc {
	var leadColon string
	if idx > 0 {
		leadColon = ":"
	}
	substr := []byte(leadColon + value + ":")
	return func(line []byte) (v any, err error) {
		if !bytes.Contains(line, substr) || bytes.Count(line, colon) < 3 {
			return
		}
		// wheel:*:0:root
		parts := strings.SplitN(string(line), ":", 4)
		if len(parts) < 4 || parts[0] == "" || parts[idx] != value ||
			// If the file contains +foo and you search for "foo", glibc
			// returns an "invalid argument" error. Similarly, if you search
			// for a gid for a row where the group name starts with "+" or "-",
			// glibc fails to find the record.
			parts[0][0] == '+' || parts[0][0] == '-' {
			return
		}
		if _, err := strconv.Atoi(parts[2]); err != nil {
			return nil, nil
		}
		return &Group{Name: parts[0], Gid: parts[2]}, nil
	}
}

func findGroupId(id string, r io.Reader) (*Group, error) {
	if v, err := readColonFile(r, matchGroupIndexValue(id, 2), 3); err != nil {
		return nil, err
	} else if v != nil {
		return v.(*Group), nil
	}
	return nil, UnknownGroupIdError(id)
}

func findGroupName(name string, r io.Reader) (*Group, error) {
	if v, err := readColonFile(r, matchGroupIndexValue(name, 0), 3); err != nil {
		return nil, err
	} else if v != nil {
		return v.(*Group), nil
	}
	return nil, UnknownGroupError(name)
}

// returns a *User for a row if that row's has the given value at the
// given index.
func matchUserIndexValue(value string, idx int) lineFunc {
	var leadColon string
	if idx > 0 {
		leadColon = ":"
	}
	substr := []byte(leadColon + value + ":")
	return func(line []byte) (v any, err error) {
		if !bytes.Contains(line, substr) || bytes.Count(line, colon) < 6 {
			return
		}
		// kevin:x:1005:1006::/home/kevin:/usr/bin/zsh
		parts := strings.SplitN(string(line), ":", 7)
		if len(parts) < 6 || parts[idx] != value || parts[0] == "" ||
			parts[0][0] == '+' || parts[0][0] == '-' {
			return
		}
		if _, err := strconv.Atoi(parts[2]); err != nil {
			return nil, nil
		}
		if _, err := strconv.Atoi(parts[3]); err != nil {
			return nil, nil
		}
		u := &User{
			Username: parts[0],
			Uid:      parts[2],
			Gid:      parts[3],
			Name:     parts[4],
			HomeDir:  parts[5],
		}
		// The pw_gecos field isn't quite standardized. Some docs
		// say: "It is expected to be a comma separated list of
		// personal data where the first item is the full name of the
		// user."
		u.Name, _, _ = strings.Cut(u.Name, ",")
		return u, nil
	}
}

func findUserId(uid string, r io.Reader) (*User, error) {
	i, e := strconv.Atoi(uid)
	if e != nil {
		return nil, errors.New("user: invalid userid " + uid)
	}
	if v, err := readColonFile(r, matchUserIndexValue(uid, 2), 6); err != nil {
		return nil, err
	} else if v != nil {
		return v.(*User), nil
	}
	return nil, UnknownUserIdError(i)
}

func findUsername(name string, r io.Reader) (*User, error) {
	if v, err := readColonFile(r, matchUserIndexValue(name, 0), 6); err != nil {
		return nil, err
	} else if v != nil {
		return v.(*User), nil
	}
	return nil, UnknownUserError(name)
}

func lookupGroup(groupname string) (*Group, error) {
	f, err := os.Open(groupFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return findGroupName(groupname, f)
}

func lookupGroupId(id string) (*Group, error) {
	f, err := os.Open(groupFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return findGroupId(id, f)
}

func lookupUser(username string) (*User, error) {
	f, err := os.Open(userFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return findUsername(username, f)
}

func lookupUserId(uid string) (*User, error) {
	f, err := os.Open(userFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return findUserId(uid, f)
}
