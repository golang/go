package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	gox "github.com/alfu32/go-shared/gox"
)

func goTool() (string, []string) {
	tool := os.Getenv("GOX_GO")
	if tool == "" {
		tool = filepath.Join(runtime.GOROOT(), "bin", "go")
		if runtime.GOOS == "windows" {
			tool += ".exe"
		}
	}
	if strings.HasSuffix(tool, ".py") {
		python := os.Getenv("GOX_PYTHON")
		if python == "" {
			python = "python"
		}
		return python, []string{tool}
	}
	return tool, nil
}

// The Go command supplies module/workspace and replacement resolution; GoX does
// not maintain a second parser for go.mod or alter module manifests.
func prepareWorkspace(fallback string, opts gox.ProjectOptions) (gox.Overlay, error) {
	tool, prefix := goTool()
	cmd := exec.Command(tool, append(prefix, "list", "-m", "-json", "all")...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	b, e := cmd.Output()
	if e != nil && strings.Contains(stderr.String(), "vendor") {
		cmd = exec.Command(tool, append(prefix, "list", "-m", "-json")...)
		stderr.Reset()
		cmd.Stderr = &stderr
		b, e = cmd.Output()
	}
	if e != nil {
		// Outside module mode, a single local GOPATH tree is still supported.
		if _, err := os.Stat(filepath.Join(fallback, "go.mod")); err != nil {
			return gox.PrepareOverlay(fallback, opts)
		}
		return gox.Overlay{}, fmt.Errorf("module discovery: %w: %s", e, stderr.String())
	}
	out := gox.Overlay{Replace: map[string]string{}}
	seen := map[string]bool{}
	dec := json.NewDecoder(bytes.NewReader(b))
	for {
		var mod struct {
			Dir     string
			Main    bool
			Replace *struct{ Dir, Version string }
		}
		if e := dec.Decode(&mod); e == io.EOF {
			break
		} else if e != nil {
			return out, e
		}
		dir := mod.Dir
		if mod.Replace != nil && mod.Replace.Dir != "" {
			dir = mod.Replace.Dir
		}
		if dir == "" || seen[dir] {
			continue
		}
		seen[dir] = true
		// Versioned module sources are immutable but overlays can safely transform
		// them too. Runtime dependencies must already be in the user's module graph.
		o, e := gox.PrepareOverlay(dir, opts)
		if e != nil {
			return out, e
		}
		for k, v := range o.Replace {
			out.Replace[k] = v
		}
		if mod.Main {
			vendor := filepath.Join(dir, "vendor")
			if info, e := os.Stat(vendor); e == nil && info.IsDir() {
				o, e := gox.PrepareOverlay(vendor, opts)
				if e != nil {
					return out, e
				}
				for k, v := range o.Replace {
					out.Replace[k] = v
				}
			}
		}
	}
	return out, nil
}
