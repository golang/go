// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd !android,linux nacl netbsd openbsd solaris
// +build !cgo

package user

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"strings"
)

const groupFile = "/etc/group"

var colon = []byte{':'}

func init() {
	groupImplemented = false
}

func findGroupId(id string, r io.Reader) (*Group, error) {
	bs := bufio.NewScanner(r)
	substr := []byte(":" + id)
	for bs.Scan() {
		lineBytes := bs.Bytes()
		if !bytes.Contains(lineBytes, substr) || bytes.Count(lineBytes, colon) < 3 {
			continue
		}
		text := strings.TrimSpace(removeComment(string(lineBytes)))
		// wheel:*:0:root
		parts := strings.SplitN(text, ":", 4)
		if len(parts) < 4 {
			continue
		}
		if parts[2] == id {
			return &Group{Name: parts[0], Gid: parts[2]}, nil
		}
	}
	if err := bs.Err(); err != nil {
		return nil, err
	}
	return nil, UnknownGroupIdError(id)
}

func findGroupName(name string, r io.Reader) (*Group, error) {
	bs := bufio.NewScanner(r)
	substr := []byte(name + ":")
	for bs.Scan() {
		lineBytes := bs.Bytes()
		if !bytes.Contains(lineBytes, substr) || bytes.Count(lineBytes, colon) < 3 {
			continue
		}
		text := strings.TrimSpace(removeComment(string(lineBytes)))
		// wheel:*:0:root
		parts := strings.SplitN(text, ":", 4)
		if len(parts) < 4 {
			continue
		}
		if parts[0] == name && parts[2] != "" {
			return &Group{Name: parts[0], Gid: parts[2]}, nil
		}
	}
	if err := bs.Err(); err != nil {
		return nil, err
	}
	return nil, UnknownGroupError(name)
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

// removeComment returns line, removing any '#' byte and any following
// bytes.
func removeComment(line string) string {
	if i := strings.Index(line, "#"); i != -1 {
		return line[:i]
	}
	return line
}
