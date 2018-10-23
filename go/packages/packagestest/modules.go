// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest

import (
	"archive/zip"
	"bytes"
	"fmt"
	"golang.org/x/tools/go/packages"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
)

// Modules is the exporter that produces module layouts.
// Each "repository" is put in it's own module, and the module file generated
// will have replace directives for all other modules.
// Given the two files
//     golang.org/repoa#a/a.go
//     golang.org/repob#b/b.go
// You would get the directory layout
//     /sometemporarydirectory
//     ├── repoa
//     │   ├── a
//     │   │   └── a.go
//     │   └── go.mod
//     └── repob
//         ├── b
//         │   └── b.go
//         └── go.mod
// and the working directory would be
//     /sometemporarydirectory/repoa
var Modules = modules{}

const theVersion = "v1.0.0"

type modules struct{}

func (modules) Name() string {
	return "Modules"
}

func (modules) Filename(exported *Exported, module, fragment string) string {
	if module == exported.primary {
		return filepath.Join(primaryDir(exported), fragment)
	}
	return filepath.Join(moduleDir(exported, module), fragment)
}

func (modules) Finalize(exported *Exported) error {
	// Write out the primary module. This module can use symlinks and
	// other weird stuff, and will be the working dir for the go command.
	// It depends on all the other modules.
	primaryDir := primaryDir(exported)
	exported.Config.Dir = primaryDir
	exported.written[exported.primary]["go.mod"] = filepath.Join(primaryDir, "go.mod")
	primaryGomod := "module " + exported.primary + "\nrequire (\n"
	for other := range exported.written {
		if other == exported.primary {
			continue
		}
		primaryGomod += fmt.Sprintf("\t%v %v\n", other, theVersion)
	}
	primaryGomod += ")\n"
	if err := ioutil.WriteFile(filepath.Join(primaryDir, "go.mod"), []byte(primaryGomod), 0644); err != nil {
		return err
	}

	// Create the mod cache so we can rename it later, even if we don't need it.
	if err := os.MkdirAll(modCache(exported), 0755); err != nil {
		return err
	}

	// Write out the go.mod files for the other modules.
	for module, files := range exported.written {
		if module == exported.primary {
			continue
		}
		dir := moduleDir(exported, module)

		modfile := filepath.Join(dir, "go.mod")
		if err := ioutil.WriteFile(modfile, []byte("module "+module+"\n"), 0644); err != nil {
			return err
		}
		files["go.mod"] = modfile
	}

	// Zip up all the secondary modules into the proxy dir.
	proxyDir := filepath.Join(exported.temp, "modproxy")
	for module, files := range exported.written {
		if module == exported.primary {
			continue
		}
		dir := filepath.Join(proxyDir, module, "@v")

		if err := writeModuleProxy(dir, module, files); err != nil {
			return fmt.Errorf("creating module proxy dir for %v: %v", module, err)
		}
	}

	// Discard the original mod cache dir, which contained the files written
	// for us by Export.
	if err := os.Rename(modCache(exported), modCache(exported)+".orig"); err != nil {
		return err
	}
	exported.Config.Env = append(exported.Config.Env,
		"GO111MODULE=on",
		"GOPATH="+filepath.Join(exported.temp, "modcache"),
		"GOPROXY=file://"+filepath.ToSlash(proxyDir))

	// Run go mod download to recreate the mod cache dir with all the extra
	// stuff in cache. All the files created by Export should be recreated.
	if err := invokeGo(exported.Config, "mod", "download"); err != nil {
		return err
	}

	return nil
}

// writeModuleProxy creates a directory in the proxy dir for a module.
func writeModuleProxy(dir, module string, files map[string]string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// list file. Just the single version.
	if err := ioutil.WriteFile(filepath.Join(dir, "list"), []byte(theVersion+"\n"), 0644); err != nil {
		return err
	}

	// go.mod, copied from the file written in Finalize.
	modContents, err := ioutil.ReadFile(files["go.mod"])
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(dir, theVersion+".mod"), modContents, 0644); err != nil {
		return err
	}

	// info file, just the bare bones.
	infoContents := []byte(fmt.Sprintf(`{"Version": "%v", "Time":"2017-12-14T13:08:43Z"}`, theVersion))
	if err := ioutil.WriteFile(filepath.Join(dir, theVersion+".info"), infoContents, 0644); err != nil {
		return err
	}

	// zip of all the source files.
	f, err := os.OpenFile(filepath.Join(dir, theVersion+".zip"), os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	z := zip.NewWriter(f)
	for name, path := range files {
		zf, err := z.Create(module + "@" + theVersion + "/" + name)
		if err != nil {
			return err
		}
		contents, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}
		if _, err := zf.Write(contents); err != nil {
			return err
		}
	}
	if err := z.Close(); err != nil {
		return err
	}

	return nil
}

func invokeGo(cfg *packages.Config, args ...string) error {
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)
	cmd := exec.Command("go", args...)
	cmd.Env = append(append([]string{}, cfg.Env...), "PWD="+cfg.Dir)
	cmd.Dir = cfg.Dir
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("go %v: %s: %s", args, err, stderr)
	}
	return nil
}

func modCache(exported *Exported) string {
	return filepath.Join(exported.temp, "modcache/pkg/mod")
}

func primaryDir(exported *Exported) string {
	return filepath.Join(exported.temp, "primarymod", path.Base(exported.primary))
}

func moduleDir(exported *Exported, module string) string {
	return filepath.Join(modCache(exported), path.Dir(module), path.Base(module)+"@"+theVersion)
}
