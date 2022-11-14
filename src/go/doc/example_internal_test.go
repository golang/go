// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"go/parser"
	"go/token"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestImportGroupStarts(t *testing.T) {
	for _, test := range []struct {
		name string
		in   string
		want []string // paths of group-starting imports
	}{
		{
			name: "one group",
			in: `package p
import (
	"a"
	"b"
	"c"
	"d"
)
`,
			want: []string{"a"},
		},
		{
			name: "several groups",
			in: `package p
import (
	"a"

	"b"
	"c"

	"d"
)
`,
			want: []string{"a", "b", "d"},
		},
		{
			name: "extra space",
			in: `package p
import (
	"a"


	"b"
	"c"


	"d"
)
`,
			want: []string{"a", "b", "d"},
		},
		{
			name: "line comment",
			in: `package p
import (
	"a" // comment
	"b" // comment

	"c"
)`,
			want: []string{"a", "c"},
		},
		{
			name: "named import",
			in: `package p
import (
	"a"
	n "b"

	m "c"
	"d"
)`,
			want: []string{"a", "c"},
		},
		{
			name: "blank import",
			in: `package p
import (
	"a"

	_ "b"

	_ "c"
	"d"
)`,
			want: []string{"a", "b", "c"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "test.go", strings.NewReader(test.in), parser.ParseComments)
			if err != nil {
				t.Fatal(err)
			}
			imps := findImportGroupStarts1(file.Imports)
			got := make([]string, len(imps))
			for i, imp := range imps {
				got[i], err = strconv.Unquote(imp.Path.Value)
				if err != nil {
					t.Fatal(err)
				}
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("got %v, want %v", got, test.want)
			}
		})
	}

}
