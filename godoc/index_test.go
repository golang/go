// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"reflect"
	"strings"
	"testing"

	"code.google.com/p/go.tools/godoc/vfs/mapfs"
)

func TestIndex(t *testing.T) {
	c := NewCorpus(mapfs.New(map[string]string{
		"src/pkg/foo/foo.go": `// Package foo is an example.
package foo

// Foo is stuff.
type Foo struct{}

func New() *Foo {
   return new(Foo)
}
`,
		"src/pkg/bar/bar.go": `// Package bar is another example to test races.
package bar
`,
		"src/pkg/skip/skip.go": `// Package skip should be skipped.
package skip
func Skip() {}
`,
	}))
	c.IndexEnabled = true
	c.IndexDirectory = func(dir string) bool {
		return !strings.Contains(dir, "skip")
	}

	if err := c.Init(); err != nil {
		t.Fatal(err)
	}
	c.UpdateIndex()
	ix, _ := c.CurrentIndex()
	if ix == nil {
		t.Fatal("no index")
	}
	t.Logf("Got: %#v", ix)
	wantStats := Statistics{Bytes: 179, Files: 2, Lines: 11, Words: 5, Spots: 7}
	if !reflect.DeepEqual(ix.Stats(), wantStats) {
		t.Errorf("Stats = %#v; want %#v", ix.Stats(), wantStats)
	}
	if _, ok := ix.words["Skip"]; ok {
		t.Errorf("the word Skip was found; expected it to be skipped")
	}
}
