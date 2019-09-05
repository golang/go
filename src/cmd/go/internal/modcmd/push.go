// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod push

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modpush"
	"fmt"
)

var cmdPush = &base.Command{
	UsageLine: "go mod push [-project <vcs>/<owner>/<name> -file project.zip]",
	Short:     "pushes the project to a hosted repository",
	Long: `
Pushes a project zip to a configured host, the environment variable GOPUSH is used 
to determine the repository endpoint or it can be overridden using -host parameter

Required fields are:
	-project which is in the the form of <vcs>/owner>/<name>, for example github.com/golang/protobuf
	-file filename of the package to build uploaded, this is in the form of SEMVER.zip for example 'v1.0.0.zip', this file can be generated using 'go mod pack'

To upload using basic authentication use -u and -p for username and password respectively
`,
	Run: runPush,
}

var (
	host     string
	project  string
	filename string
	username string
	password string
)

func init() {
	cmdPush.Flag.StringVar(&host, "host", cfg.GOPUSH, "Overrides the GOPUSH environment setting")
	cmdPush.Flag.StringVar(&project, "project", "", "in the form <vcs>/<owner>/<name>")
	cmdPush.Flag.StringVar(&filename, "file", "", "filename of the package to be uploaded")
	cmdPush.Flag.StringVar(&username, "u", "", "username for basic authentication")
	cmdPush.Flag.StringVar(&password, "p", "", "password for basic authentication")
}

func runPush(cmd *base.Command, args []string) {
	if project == "" {
		base.Fatalf("project must be specified in the form <vcs>/<owner>/<name> (use -project)")
	}
	if filename == "" {
		base.Fatalf("file must be specified (use -file")
	}
	if host == "" {
		base.Fatalf("Host must be set, either use the environment variable 'GOPUSH' or use the -host flag")
	}

	fmt.Printf("Pushing file %s to project %s at %s\n", filename, project, host)
	modpush.Push(host, project, filename, username, password)
}
