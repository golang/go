// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"sort"
	"testing"

	"golang.org/x/tools/internal/span"
)

func (r *runner) References(t *testing.T, spn span.Span, itemList []span.Span) {
	for _, includeDeclaration := range []bool{true, false} {
		t.Run(fmt.Sprintf("refs-declaration-%v", includeDeclaration), func(t *testing.T) {
			var itemStrings []string
			for i, s := range itemList {
				// We don't want the first result if we aren't including the declaration.
				if i == 0 && !includeDeclaration {
					continue
				}
				itemStrings = append(itemStrings, fmt.Sprint(s))
			}
			sort.Strings(itemStrings)
			var expect string
			for _, s := range itemStrings {
				expect += s + "\n"
			}
			expect = r.Normalize(expect)

			uri := spn.URI()
			filename := uri.Filename()
			target := filename + fmt.Sprintf(":%v:%v", spn.Start().Line(), spn.Start().Column())
			args := []string{"references"}
			if includeDeclaration {
				args = append(args, "-d")
			}
			args = append(args, target)
			got, stderr := r.NormalizeGoplsCmd(t, args...)
			if stderr != "" {
				t.Errorf("references failed for %s: %s", target, stderr)
			} else if expect != got {
				t.Errorf("references failed for %s expected:\n%s\ngot:\n%s", target, expect, got)
			}
		})
	}
}
