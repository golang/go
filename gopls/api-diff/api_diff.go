// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/source"
	diffpkg "golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/gocommand"
)

var (
	previousVersionFlag = flag.String("prev", "", "version to compare against")
	versionFlag         = flag.String("version", "", "version being tagged, or current version if omitted")
)

func main() {
	flag.Parse()

	apiDiff, err := diffAPI(*versionFlag, *previousVersionFlag)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf(`
%s
`, apiDiff)
}

type JSON interface {
	String() string
	Write(io.Writer)
}

func diffAPI(version, prev string) (string, error) {
	ctx := context.Background()
	previousApi, err := loadAPI(ctx, prev)
	if err != nil {
		return "", fmt.Errorf("load previous API: %v", err)
	}
	var currentApi *source.APIJSON
	if version == "" {
		currentApi = source.GeneratedAPIJSON
	} else {
		var err error
		currentApi, err = loadAPI(ctx, version)
		if err != nil {
			return "", fmt.Errorf("load current API: %v", err)
		}
	}

	b := &strings.Builder{}
	if err := diff(b, previousApi.Commands, currentApi.Commands, "command", func(c *source.CommandJSON) string {
		return c.Command
	}, diffCommands); err != nil {
		return "", fmt.Errorf("diff commands: %v", err)
	}
	if diff(b, previousApi.Analyzers, currentApi.Analyzers, "analyzer", func(a *source.AnalyzerJSON) string {
		return a.Name
	}, diffAnalyzers); err != nil {
		return "", fmt.Errorf("diff analyzers: %v", err)
	}
	if err := diff(b, previousApi.Lenses, currentApi.Lenses, "code lens", func(l *source.LensJSON) string {
		return l.Lens
	}, diffLenses); err != nil {
		return "", fmt.Errorf("diff lenses: %v", err)
	}
	for key, prev := range previousApi.Options {
		current, ok := currentApi.Options[key]
		if !ok {
			panic(fmt.Sprintf("unexpected option key: %s", key))
		}
		if err := diff(b, prev, current, "option", func(o *source.OptionJSON) string {
			return o.Name
		}, diffOptions); err != nil {
			return "", fmt.Errorf("diff options (%s): %v", key, err)
		}
	}

	return b.String(), nil
}

func diff[T JSON](b *strings.Builder, previous, new []T, kind string, uniqueKey func(T) string, diffFunc func(*strings.Builder, T, T)) error {
	prevJSON := collect(previous, uniqueKey)
	newJSON := collect(new, uniqueKey)
	for k := range newJSON {
		delete(prevJSON, k)
	}
	for _, deleted := range prevJSON {
		b.WriteString(fmt.Sprintf("%s %s was deleted.\n", kind, deleted))
	}
	for _, prev := range previous {
		delete(newJSON, uniqueKey(prev))
	}
	if len(newJSON) > 0 {
		b.WriteString("The following commands were added:\n")
		for _, n := range newJSON {
			n.Write(b)
			b.WriteByte('\n')
		}
	}
	previousMap := collect(previous, uniqueKey)
	for _, current := range new {
		prev, ok := previousMap[uniqueKey(current)]
		if !ok {
			continue
		}
		c, p := bytes.NewBuffer(nil), bytes.NewBuffer(nil)
		prev.Write(p)
		current.Write(c)
		if diff := diffStr(p.String(), c.String()); diff != "" {
			diffFunc(b, prev, current)
			b.WriteString("\n--\n")
		}
	}
	return nil
}

func collect[T JSON](args []T, uniqueKey func(T) string) map[string]T {
	m := map[string]T{}
	for _, arg := range args {
		m[uniqueKey(arg)] = arg
	}
	return m
}

var goCmdRunner = gocommand.Runner{}

func loadAPI(ctx context.Context, version string) (*source.APIJSON, error) {
	tmpGopath, err := ioutil.TempDir("", "gopath*")
	if err != nil {
		return nil, fmt.Errorf("temp dir: %v", err)
	}
	defer os.RemoveAll(tmpGopath)

	exampleDir := fmt.Sprintf("%s/src/example.com", tmpGopath)
	if err := os.MkdirAll(exampleDir, 0776); err != nil {
		return nil, fmt.Errorf("mkdir: %v", err)
	}

	if stdout, err := goCmdRunner.Run(ctx, gocommand.Invocation{
		Verb:       "mod",
		Args:       []string{"init", "example.com"},
		WorkingDir: exampleDir,
		Env:        append(os.Environ(), fmt.Sprintf("GOPATH=%s", tmpGopath)),
	}); err != nil {
		return nil, fmt.Errorf("go mod init failed: %v (stdout: %v)", err, stdout)
	}
	if stdout, err := goCmdRunner.Run(ctx, gocommand.Invocation{
		Verb:       "install",
		Args:       []string{fmt.Sprintf("golang.org/x/tools/gopls@%s", version)},
		WorkingDir: exampleDir,
		Env:        append(os.Environ(), fmt.Sprintf("GOPATH=%s", tmpGopath)),
	}); err != nil {
		return nil, fmt.Errorf("go install failed: %v (stdout: %v)", err, stdout.String())
	}
	cmd := exec.Cmd{
		Path: filepath.Join(tmpGopath, "bin", "gopls"),
		Args: []string{"gopls", "api-json"},
		Dir:  tmpGopath,
	}
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("output: %v", err)
	}
	apiJson := &source.APIJSON{}
	if err := json.Unmarshal(out, apiJson); err != nil {
		return nil, fmt.Errorf("unmarshal: %v", err)
	}
	return apiJson, nil
}

func diffCommands(b *strings.Builder, prev, current *source.CommandJSON) {
	if prev.Title != current.Title {
		b.WriteString(fmt.Sprintf("Title changed from %q to %q\n", prev.Title, current.Title))
	}
	if prev.Doc != current.Doc {
		b.WriteString(fmt.Sprintf("Documentation changed from %q to %q\n", prev.Doc, current.Doc))
	}
	if prev.ArgDoc != current.ArgDoc {
		b.WriteString("Arguments changed from " + formatBlock(prev.ArgDoc) + " to " + formatBlock(current.ArgDoc))
	}
	if prev.ResultDoc != current.ResultDoc {
		b.WriteString("Results changed from " + formatBlock(prev.ResultDoc) + " to " + formatBlock(current.ResultDoc))
	}
}

func diffAnalyzers(b *strings.Builder, previous, current *source.AnalyzerJSON) {
	b.WriteString(fmt.Sprintf("Changes to analyzer %s:\n\n", current.Name))
	if previous.Doc != current.Doc {
		b.WriteString(fmt.Sprintf("Documentation changed from %q to %q\n", previous.Doc, current.Doc))
	}
	if previous.Default != current.Default {
		b.WriteString(fmt.Sprintf("Default changed from %v to %v\n", previous.Default, current.Default))
	}
}

func diffLenses(b *strings.Builder, previous, current *source.LensJSON) {
	b.WriteString(fmt.Sprintf("Changes to code lens %s:\n\n", current.Title))
	if previous.Title != current.Title {
		b.WriteString(fmt.Sprintf("Title changed from %q to %q\n", previous.Title, current.Title))
	}
	if previous.Doc != current.Doc {
		b.WriteString(fmt.Sprintf("Documentation changed from %q to %q\n", previous.Doc, current.Doc))
	}
}

func diffOptions(b *strings.Builder, previous, current *source.OptionJSON) {
	b.WriteString(fmt.Sprintf("Changes to option %s:\n\n", current.Name))
	if previous.Doc != current.Doc {
		diff := diffStr(previous.Doc, current.Doc)
		fmt.Fprintf(b, "Documentation changed:\n%s\n", diff)
	}
	if previous.Default != current.Default {
		b.WriteString(fmt.Sprintf("Default changed from %q to %q\n", previous.Default, current.Default))
	}
	if previous.Hierarchy != current.Hierarchy {
		b.WriteString(fmt.Sprintf("Categorization changed from %q to %q\n", previous.Hierarchy, current.Hierarchy))
	}
	if previous.Status != current.Status {
		b.WriteString(fmt.Sprintf("Status changed from %q to %q\n", previous.Status, current.Status))
	}
	if previous.Type != current.Type {
		b.WriteString(fmt.Sprintf("Type changed from %q to %q\n", previous.Type, current.Type))
	}
	// TODO(rstambler): Handle possibility of same number but different keys/values.
	if len(previous.EnumKeys.Keys) != len(current.EnumKeys.Keys) {
		b.WriteString(fmt.Sprintf("Enum keys changed from\n%s\n to \n%s\n", previous.EnumKeys, current.EnumKeys))
	}
	if len(previous.EnumValues) != len(current.EnumValues) {
		b.WriteString(fmt.Sprintf("Enum values changed from\n%s\n to \n%s\n", previous.EnumValues, current.EnumValues))
	}
}

func formatBlock(str string) string {
	if str == "" {
		return `""`
	}
	return "\n```\n" + str + "\n```\n"
}

func diffStr(before, after string) string {
	if before == after {
		return ""
	}
	// Add newlines to avoid newline messages in diff.
	unified := diffpkg.Unified("previous", "current", before+"\n", after+"\n")
	return fmt.Sprintf("%q", unified)
}
