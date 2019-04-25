// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package snippet

import (
	"testing"
)

func TestSnippetBuilder(t *testing.T) {
	expect := func(expected string, fn func(*Builder)) {
		var b Builder
		fn(&b)
		if got := b.String(); got != expected {
			t.Errorf("got %q, expected %q", got, expected)
		}
	}

	expect("", func(b *Builder) {})

	expect(`hi { \} \$ | " , / \\`, func(b *Builder) {
		b.WriteText(`hi { } $ | " , / \`)
	})

	expect("${1}", func(b *Builder) {
		b.WritePlaceholder(nil)
	})

	expect("hi ${1:there}", func(b *Builder) {
		b.WriteText("hi ")
		b.WritePlaceholder(func(b *Builder) {
			b.WriteText("there")
		})
	})

	expect(`${1:id=${2:{your id\}}}`, func(b *Builder) {
		b.WritePlaceholder(func(b *Builder) {
			b.WriteText("id=")
			b.WritePlaceholder(func(b *Builder) {
				b.WriteText("{your id}")
			})
		})
	})

	expect(`${1|one,{ \} \$ \| " \, / \\,three|}`, func(b *Builder) {
		b.WriteChoice([]string{"one", `{ } $ | " , / \`, "three"})
	})
}
