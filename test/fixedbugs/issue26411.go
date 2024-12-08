// run

//go:build !nacl && !js && !wasip1

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

	tests := []struct {
		code   string
		errors []string
	}{
		{
			code: `
package main

func main() {
foo:
foo:
}
`,
			errors: []string{
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

			errors: []string{
				"^.+:6:13: label bar defined and not used\n",
				".+:7:4: label bar already defined at .+:6:13\n",
				".+:8:1: label bar already defined at .+:6:13\n",
				".+:9:1: label bar already defined at .+:6:13\n$",
			},
		},
	}

	for i, test := range tests {
		filename := filepath.Join(tmpdir, fmt.Sprintf("%d.go", i))
		if err := ioutil.WriteFile(filename, []byte(test.code), 0644); err != nil {
			log.Printf("#%d: failed to create file %s", i, filename)
			continue
		}
		output, _ := exec.Command("go", "tool", "compile", "-p=p", filename).CombinedOutput()

		// remove each matching error from the output
		for _, err := range test.errors {
			rx := regexp.MustCompile(err)
			match := rx.Find(output)
			output = bytes.Replace(output, match, nil, 1) // remove match (which might be nil) from output
		}

		// at this point all output should have been consumed
		if len(output) != 0 {
			log.Printf("Test case %d has unmatched errors:\n%s", i, output)
		}
	}
}
