// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
)

func TestCommentMap(t *testing.T) {
	const src = `/* ERROR "0:0" */ /* ERROR "0:0" */ // ERROR "0:0"
// ERROR "0:0"
x /* ERROR "3:1" */                // ignore automatically inserted semicolon here
/* ERROR "3:1" */                  // position of x on previous line
   x /* ERROR "5:4" */ ;           // do not ignore this semicolon
/* ERROR "5:24" */                 // position of ; on previous line
	package /* ERROR "7:2" */  // indented with tab
        import  /* ERROR "8:9" */  // indented with blanks
`
	m := CommentMap(strings.NewReader(src), regexp.MustCompile("^ ERROR "))
	found := 0 // number of errors found
	for line, errlist := range m {
		for _, err := range errlist {
			if err.Pos.Line() != line {
				t.Errorf("%v: got map line %d; want %d", err, err.Pos.Line(), line)
				continue
			}
			// err.Pos.Line() == line

			got := strings.TrimSpace(err.Msg[len(" ERROR "):])
			want := fmt.Sprintf(`"%d:%d"`, line, err.Pos.Col())
			if got != want {
				t.Errorf("%v: got msg %q; want %q", err, got, want)
				continue
			}
			found++
		}
	}

	want := strings.Count(src, " ERROR ")
	if found != want {
		t.Errorf("CommentMap got %d errors; want %d", found, want)
	}
}
