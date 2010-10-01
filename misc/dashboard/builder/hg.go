package main

import (
	"fmt"
	"os"
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
	s, _, err := runLog(nil, goroot,
		"hg", "log", "-r", rev, "-l", "1", "--template", format)
	if err != nil {
		return
	}
	return strings.Split(s, ">", 5), nil
}
