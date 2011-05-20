// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"testing"
)

var userIdTests = []struct {
	id                   string
	name, comment, email string
}{
	{"", "", "", ""},
	{"John Smith", "John Smith", "", ""},
	{"John Smith ()", "John Smith", "", ""},
	{"John Smith () <>", "John Smith", "", ""},
	{"(comment", "", "comment", ""},
	{"(comment)", "", "comment", ""},
	{"<email", "", "", "email"},
	{"<email>   sdfk", "", "", "email"},
	{"  John Smith  (  Comment ) asdkflj < email > lksdfj", "John Smith", "Comment", "email"},
	{"  John Smith  < email > lksdfj", "John Smith", "", "email"},
	{"(<foo", "", "<foo", ""},
	{"René Descartes (العربي)", "René Descartes", "العربي", ""},
}

func TestParseUserId(t *testing.T) {
	for i, test := range userIdTests {
		name, comment, email := parseUserId(test.id)
		if name != test.name {
			t.Errorf("%d: name mismatch got:%s want:%s", i, name, test.name)
		}
		if comment != test.comment {
			t.Errorf("%d: comment mismatch got:%s want:%s", i, comment, test.comment)
		}
		if email != test.email {
			t.Errorf("%d: email mismatch got:%s want:%s", i, email, test.email)
		}
	}
}

var newUserIdTests = []struct {
	name, comment, email, id string
}{
	{"foo", "", "", "foo"},
	{"", "bar", "", "(bar)"},
	{"", "", "baz", "<baz>"},
	{"foo", "bar", "", "foo (bar)"},
	{"foo", "", "baz", "foo <baz>"},
	{"", "bar", "baz", "(bar) <baz>"},
	{"foo", "bar", "baz", "foo (bar) <baz>"},
}

func TestNewUserId(t *testing.T) {
	for i, test := range newUserIdTests {
		uid := NewUserId(test.name, test.comment, test.email)
		if uid == nil {
			t.Errorf("#%d: returned nil", i)
			continue
		}
		if uid.Id != test.id {
			t.Errorf("#%d: got '%s', want '%s'", i, uid.Id, test.id)
		}
	}
}

var invalidNewUserIdTests = []struct {
	name, comment, email string
}{
	{"foo(", "", ""},
	{"foo<", "", ""},
	{"", "bar)", ""},
	{"", "bar<", ""},
	{"", "", "baz>"},
	{"", "", "baz)"},
	{"", "", "baz\x00"},
}

func TestNewUserIdWithInvalidInput(t *testing.T) {
	for i, test := range invalidNewUserIdTests {
		if uid := NewUserId(test.name, test.comment, test.email); uid != nil {
			t.Errorf("#%d: returned non-nil value: %#v", i, uid)
		}
	}
}
