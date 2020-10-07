// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"strings"
	"testing"
)

func TestErrorMap(t *testing.T) {
	src := strings.NewReader(
		`/* ERROR 1:1 */ /* ERROR "1:1" */ // ERROR 1:1
// ERROR "1:1"
x /* ERROR 3:1 */                // ignore automatically inserted semicolon here
/* ERROR 3:1 */                  // position of x on previous line
   x /* ERROR 5:4 */ ;           // do not ignore this semicolon
/* ERROR 5:22 */                 // position of ; on previous line
	package /* ERROR 7:2 */  // indented with tab
        import  /* ERROR 8:9 */  // indented with blanks
`)
	m := ErrorMap(src)
	for line, errlist := range m {
		for _, err := range errlist {
			if err.Pos.Line() != line {
				t.Errorf("%v: got map line %d; want %d", err, err.Pos.Line(), line)
				continue
			}
			// err.Pos.Line() == line
			msg := fmt.Sprintf("%d:%d", line, err.Pos.Col())
			if err.Msg != msg {
				t.Errorf("%v: got msg %q; want %q", err, err.Msg, msg)
				continue
			}
		}
	}
}
