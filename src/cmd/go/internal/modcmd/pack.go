// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod pack

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/modpack"
	"os"
)

var cmdPack = &base.Command{
	UsageLine: "go mod pack [<module> <version>]",
	Short:     "packages the project in the current location",
	Long: `
Packages a project in the current directory,
The package created can then be uploaded to a host compatible
server.
`,
	Run: runPack,
}

func runPack(cmd *base.Command, args []string) {

	if len(args) != 2 {
		base.Fatalf("go mod pack: no. of arguments incorrect")
	}

	if err := modpack.ValidateArgument(args); err != nil {
		base.Fatalf("go mod pack: argument %v invalid: %v", args[0], err)
	}

	if _, err := os.Stat("go.mod"); err != nil {
		base.Fatalf("go mod pack: go.mod does not exists")
	}

	modpack.Pack(args) // does all the hard work
}
