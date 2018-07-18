// +build !nacl,!js
// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that label redefinition errors print out
// a column number that matches the start of the current label's
// definition instead of the label delimiting token ":"

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

func main() {
	tmpdir, err := ioutil.TempDir("", "issue26411")
	if err != nil {
		log.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	samples := []struct {
		code       string
		wantOutput []string
	}{
		{
			code: `
package main

func main() {
foo:
foo:
}
`,
			wantOutput: []string{
				"^.+:5:1: label foo defined and not used\n",
				".+:6:1: label foo already defined at .+:5:1\n$",
			},
		},
		{
			code: `
package main

func main() {

            bar:
   bar:
bar:
bar            :
}
`,

			wantOutput: []string{
				"^.+:6:13: label bar defined and not used\n",
				".+:7:4: label bar already defined at .+:6:13\n",
				".+:8:1: label bar already defined at .+:6:13\n",
				".+:9:1: label bar already defined at .+:6:13\n$",
			},
		},
	}

	for i, sample := range samples {
		filename := filepath.Join(tmpdir, fmt.Sprintf("%d.go", i))
		if err := ioutil.WriteFile(filename, []byte(sample.code), 0644); err != nil {
			log.Printf("#%d: failed to create file %s", i, filename)
			continue
		}
		output, _ := exec.Command("go", "tool", "compile", filename).CombinedOutput()

		// Now match the output
		for _, regex := range sample.wantOutput {
			reg := regexp.MustCompile(regex)
			matches := reg.FindAll(output, -1)
			for _, match := range matches {
				index := bytes.Index(output, match)
				output = bytes.Join([][]byte{output[:index], output[index+len(match):]}, []byte(""))
			}
		}

		if len(output) != 0 {
			log.Printf("#%d: did not match all the output\nResidual output:\n\t%s", i, output)
		}
	}
}
