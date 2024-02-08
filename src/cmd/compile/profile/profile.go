// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// This script collects a CPU profile of the compiler
// for building all targets in std and cmd, and puts
// the profile at $GOROOT/src/cmd/compile/default.pgo.

var goroot = func() string {
	cmd := exec.Command("go", "env", "GOROOT")
	goroot, err := cmd.CombinedOutput()
	if err != nil {
		panic(err)
	}
	// return and remove \n
	return string(goroot[:len(goroot)-1])
}()

var hashmap = make(map[string]string) // key hash, value package name

func main() {
	var tmpdir string
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		panic(err)
	}
	olddir, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	defer func() {
		err := os.Chdir(olddir)
		if err != nil {
			panic(err)
		}
		err = os.RemoveAll(tmpdir)
		if err != nil {
			panic(err)
		}
	}()
	os.Chdir(tmpdir)

	cmd := exec.Command("go", "list", "std", "cmd")
	list, err := cmd.CombinedOutput()
	if err != nil {
		panic(err)
	}
	plist := bytes.Split(list, []byte("\n"))
	var wg sync.WaitGroup
	sema := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i := range plist {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			sema <- struct{}{}

			hash := hash(plist[i])
			pname := string(plist[i])
			fmt.Println(pname, hash)

			other, have := hashmap[hash]
			// prevents hash conflicts resulting in unexpected requests to write pprof
			// from different packages to the same file
			if have {
				panic(fmt.Sprintf("%s and %s shard hash %s\n", other, pname, hash))
			}
			hashmap[hash] = pname

			cmd := exec.Command("go", "build", "-o", os.DevNull, `-gcflags=-cpuprofile=`+filepath.Join(tmpdir, "/prof."+hash), pname)
			output, err := cmd.CombinedOutput()
			if err != nil {
				output := string(output)
				fmt.Println(output)
				// make sure don't stay silent when something unexpected happens
				if !strings.Contains(output, "no non-test Go files") &&
					!strings.Contains(output, "build constraints exclude all Go files") &&
					!strings.Contains(output, "no Go files") {
					panic(err)
				}
			}

			<-sema
		}()
	}
	wg.Wait()

	// go tool pprof -proto prof.* > $(go env GOROOT)/src/cmd/compile/default.pgo
	args := []string{"tool", "pprof", "-proto"}
	err = filepath.WalkDir(tmpdir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == tmpdir {
			return nil
		}
		if d.IsDir() {
			panic("Unexpected directory")
		}
		args = append(args, path)
		return nil
	})
	if err != nil {
		panic(err)
	}

	cmd = exec.Command("go", args...)
	result, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(result))
		panic(err)
	}
	err = os.WriteFile(filepath.Join(goroot, "src", "cmd", "compile", "default.pgo"), result, 0777)
	if err != nil {
		panic(err)
	}
}

func hash(s []byte) string {
	Md5 := md5.New()
	seed, err := time.Now().MarshalBinary()
	if err != nil {
		panic(err)
	}
	Md5.Write(seed)
	hashraw := Md5.Sum(s)
	hashresult := make([]byte, 0, len(hashraw)/6)
	tmp := 0
	maxbyte := int(^byte(0))
	// because it is use for file name, avoid generating excessively long string.
	for i := range hashraw {
		if i%6 != 0 {
			tmp += int(hashraw[i])
			continue
		}
		hashresult = append(hashresult, byte(tmp%maxbyte))
	}
	return hex.EncodeToString(hashresult)
}
