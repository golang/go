//go:build ((aix || dragonfly || freebsd || (js && wasm) || (!android && linux) || netbsd || openbsd || solaris) && (!cgo || osusergo)) || darwin

package user

// See iterate_cgo.go for explanation why this is used on darwin
// regardless of whether CGO is enabled or not.

import (
	"bytes"
	"os"
	"strconv"
	"strings"
)

// Redefining constants, because darwin can use cgo for lookup, but not for
// iterating.
const _groupFile = "/etc/group"
const _userFile = "/etc/passwd"

var _colon = []byte{':'}

func iterateUsers(fn NextUserFunc) error {
	f, err := os.Open(_userFile)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = readColonFile(f, usersIterator(fn), 6)
	return err
}

func iterateGroups(fn NextGroupFunc) error {
	f, err := os.Open(_groupFile)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = readColonFile(f, groupsIterator(fn), 3)
	return err
}

// parseGroupLine is lineFunc to parse a valid group line for iteration.
func parseGroupLine(line []byte) (v interface{}, err error) {
	if bytes.Count(line, _colon) < 3 {
		return
	}
	// wheel:*:0:root
	parts := strings.SplitN(string(line), ":", 4)
	if len(parts) < 4 || parts[0] == "" ||
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

// parseUserLine is lineFunc to parse a valid user line for iteration.
func parseUserLine(line []byte) (v interface{}, err error) {
	if bytes.Count(line, _colon) < 6 {
		return
	}
	// kevin:x:1005:1006::/home/kevin:/usr/bin/zsh
	parts := strings.SplitN(string(line), ":", 7)
	if len(parts) < 6 || parts[0] == "" ||
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
	if i := strings.Index(u.Name, ","); i >= 0 {
		u.Name = u.Name[:i]
	}
	return u, nil
}

// usersIterator parses *User and passes it to fn for each given valid line
// read by readColonFile. If non-nil error is returned from fn, iteration
// is terminated.
func usersIterator(fn NextUserFunc) lineFunc {
	return func(line []byte) (interface{}, error) {
		v, _ := parseUserLine(line)
		if u, ok := v.(*User); ok {
			if err := fn(u); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}
}

// groupsIterator parses *Group and passes it to fn for each given valid line
// read by readColonFile. If non-nil error is returned from fn, iteration
// is terminated.
func groupsIterator(fn NextGroupFunc) lineFunc {
	return func(line []byte) (interface{}, error) {
		v, _ := parseGroupLine(line)
		if g, ok := v.(*Group); ok {
			if err := fn(g); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}
}
