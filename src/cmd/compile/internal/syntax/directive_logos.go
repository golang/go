// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "strings"

// DirectiveArg reports whether text, the text of one line comment, is the
// directive whose marker is marker, such as "//logos:rodata". The marker is
// a whole word: it ends the comment or is followed by a space or a tab, so
// "//logos:rodatax" is another word and not the directive. The result is the
// text after the marker without the white space around it.
func DirectiveArg(text, marker string) (arg string, ok bool) {
	rest, ok := strings.CutPrefix(strings.TrimRight(text, " \t\r"), marker)
	if !ok || rest != "" && rest[0] != ' ' && rest[0] != '\t' {
		return "", false
	}
	return strings.TrimLeft(rest, " \t"), true
}
