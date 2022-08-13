// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"runtime"
	"strconv"

	"cmd/go/internal/base"
	"cmd/go/internal/modload"
	web "cmd/go/internal/web"

	"golang.org/x/mod/module"
)

var cmdAudit = &base.Command{
	UsageLine: "go mod audit",
	Short:     "audit dependencies for known vulnerabilities",
	Long: `
Audit checks dependencies for known vulnerabilities.
	`,
	Run: runAudit,
}

func init() {
	base.AddModCommonFlags(&cmdAudit.Flag)
}

func runAudit(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	if len(args) != 0 {
		base.Fatalf("go: verify takes no arguments")
	}
	modload.ForceUseModules = true
	modload.RootMode = modload.NeedRoot

	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))

	const defaultGoVersion = ""
	mods := modload.LoadModGraph(ctx, defaultGoVersion).BuildList()[1:]
	vulnsChans := make([]<-chan []error, len(mods))

	for i, mod := range mods {
		sem <- token{}
		vulnsc := make(chan []error, 1)
		vulnsChans[i] = vulnsc
		mod := mod
		go func() {
			vulnsc <- auditMod(mod)
			<-sem
		}()
	}

	ok := true
	vulnCount := 0
	for _, vulnsc := range vulnsChans {
		vulns := <-vulnsc
		for _, vuln := range vulns {
			vulnCount++
			base.Errorf("%s", vuln)
			ok = false
		}
	}
	if ok {
		fmt.Printf("no known vulnerabilities found\n")
	} else {
		fmt.Printf("found %s vulnerabilities\n", strconv.Itoa(vulnCount))
	}
}

func auditMod(mod module.Version) []error {
	var vulns []error
	u, err := url.Parse("https://vuln.go.dev/" + mod.Path + ".json")
	data, err := web.GetBytes(u)
	if err != nil {
		return nil
	}
	var objmap []map[string]interface{}
	data = data[:]
	if err := json.Unmarshal(data, &objmap); err != nil {
		return nil
	}
	for i := range objmap {
		aliases := objmap[i]["aliases"]
		vulns = append(vulns, fmt.Errorf("vuln found for %s: %s", mod.String(), aliases))
	}
	return vulns
}
