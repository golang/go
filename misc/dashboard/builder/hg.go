// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type Commit struct {
	num    int    // mercurial revision number
	node   string // mercurial hash
	parent string // hash of commit's parent
	user   string // author's Name <email>
	date   string // date of commit
	desc   string // description
}

// getCommit returns details about the Commit specified by the revision hash
func getCommit(rev string) (c Commit, err os.Error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("getCommit: %s: %s", rev, err)
		}
	}()
	parts, err := getCommitParts(rev)
	if err != nil {
		return
	}
	num, err := strconv.Atoi(parts[0])
	if err != nil {
		return
	}
	parent := ""
	if num > 0 {
		prev := strconv.Itoa(num - 1)
		if pparts, err := getCommitParts(prev); err == nil {
			parent = pparts[1]
		}
	}
	user := strings.Replace(parts[2], "&lt;", "<", -1)
	user = strings.Replace(user, "&gt;", ">", -1)
	return Commit{num, parts[1], parent, user, parts[3], parts[4]}, nil
}

func getCommitParts(rev string) (parts []string, err os.Error) {
	const format = "{rev}>{node}>{author|escape}>{date}>{desc}"
	s, _, err := runLog(nil, "", goroot,
		"hg", "log",
		"--encoding", "utf-8",
		"--rev", rev,
		"--limit", "1",
		"--template", format,
	)
	if err != nil {
		return
	}
	return strings.Split(s, ">", 5), nil
}

var revisionRe = regexp.MustCompile(`([0-9]+):[0-9a-f]+$`)

// getTag fetches a Commit by finding the first hg tag that matches re.
func getTag(re *regexp.Regexp) (c Commit, tag string, err os.Error) {
	o, _, err := runLog(nil, "", goroot, "hg", "tags")
	for _, l := range strings.Split(o, "\n", -1) {
		tag = re.FindString(l)
		if tag == "" {
			continue
		}
		s := revisionRe.FindStringSubmatch(l)
		if s == nil {
			err = os.NewError("couldn't find revision number")
			return
		}
		c, err = getCommit(s[1])
		return
	}
	err = os.NewError("no matching tag found")
	return
}
