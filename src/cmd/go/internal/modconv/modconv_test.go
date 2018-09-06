// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"testing"
)

var extMap = map[string]string{
	".dep":       "Gopkg.lock",
	".glide":     "glide.lock",
	".glock":     "GLOCKFILE",
	".godeps":    "Godeps/Godeps.json",
	".tsv":       "dependencies.tsv",
	".vconf":     "vendor.conf",
	".vjson":     "vendor/vendor.json",
	".vyml":      "vendor.yml",
	".vmanifest": "vendor/manifest",
}

func Test(t *testing.T) {
	tests, _ := filepath.Glob("testdata/*")
	if len(tests) == 0 {
		t.Fatalf("no tests found")
	}
	for _, test := range tests {
		file := filepath.Base(test)
		ext := filepath.Ext(file)
		if ext == ".out" {
			continue
		}
		t.Run(file, func(t *testing.T) {
			if extMap[ext] == "" {
				t.Fatal("unknown extension")
			}
			if Converters[extMap[ext]] == nil {
				t.Fatalf("Converters[%q] == nil", extMap[ext])
			}
			data, err := ioutil.ReadFile(test)
			if err != nil {
				t.Fatal(err)
			}
			out, err := Converters[extMap[ext]](test, data)
			if err != nil {
				t.Fatal(err)
			}
			want, err := ioutil.ReadFile(test[:len(test)-len(ext)] + ".out")
			if err != nil {
				t.Error(err)
			}
			var buf bytes.Buffer
			for _, r := range out.Require {
				fmt.Fprintf(&buf, "%s %s\n", r.Mod.Path, r.Mod.Version)
			}
			if !bytes.Equal(buf.Bytes(), want) {
				t.Errorf("have:\n%s\nwant:\n%s", buf.Bytes(), want)
			}
		})
	}
}
