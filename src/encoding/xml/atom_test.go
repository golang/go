// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import "time"

var atomValue = &Feed{
	XMLName: Name{"http://www.w3.org/2005/Atom", "feed"},
	Title:   "Example Feed",
	Link:    []Link{{Href: "http://example.org/"}},
	Updated: ParseTime("2003-12-13T18:30:02Z"),
	Author:  Person{Name: "John Doe"},
	Id:      "urn:uuid:60a76c80-d399-11d9-b93C-0003939e0af6",

	Entry: []Entry{
		{
			Title:   "Atom-Powered Robots Run Amok",
			Link:    []Link{{Href: "http://example.org/2003/12/13/atom03"}},
			Id:      "urn:uuid:1225c695-cfb8-4ebb-aaaa-80da344efa6a",
			Updated: ParseTime("2003-12-13T18:30:02Z"),
			Summary: NewText("Some text."),
		},
	},
}

var atomXml = `` +
	`<feed xmlns="http://www.w3.org/2005/Atom" updated="2003-12-13T18:30:02Z">` +
	`<title>Example Feed</title>` +
	`<id>urn:uuid:60a76c80-d399-11d9-b93C-0003939e0af6</id>` +
	`<link href="http://example.org/"></link>` +
	`<author><name>John Doe</name><uri></uri><email></email></author>` +
	`<entry>` +
	`<title>Atom-Powered Robots Run Amok</title>` +
	`<id>urn:uuid:1225c695-cfb8-4ebb-aaaa-80da344efa6a</id>` +
	`<link href="http://example.org/2003/12/13/atom03"></link>` +
	`<updated>2003-12-13T18:30:02Z</updated>` +
	`<author><name></name><uri></uri><email></email></author>` +
	`<summary>Some text.</summary>` +
	`</entry>` +
	`</feed>`

func ParseTime(str string) time.Time {
	t, err := time.Parse(time.RFC3339, str)
	if err != nil {
		panic(err)
	}
	return t
}

func NewText(text string) Text {
	return Text{
		Body: text,
	}
}
