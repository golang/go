// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Stripped down Atom feed data structures.

func TestUnmarshalFeed(t *testing.T) {
	var f Feed
	if err := Unmarshal([]byte(atomFeedString), &f); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if !reflect.DeepEqual(f, atomFeed) {
		t.Fatalf("have %#v\nwant %#v", f, atomFeed)
	}
}

// hget http://codereview.appspot.com/rss/mine/rsc
const atomFeedString = `
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en-us" updated="2009-10-04T01:35:58+00:00"><title>Code Review - My issues</title><link href="http://codereview.appspot.com/" rel="alternate"></link><link href="http://codereview.appspot.com/rss/mine/rsc" rel="self"></link><id>http://codereview.appspot.com/</id><author><name>rietveld&lt;&gt;</name></author><entry><title>rietveld: an attempt at pubsubhubbub
</title><link href="http://codereview.appspot.com/126085" rel="alternate"></link><updated>2009-10-04T01:35:58+00:00</updated><author><name>email-address-removed</name></author><id>urn:md5:134d9179c41f806be79b3a5f7877d19a</id><summary type="html">
  An attempt at adding pubsubhubbub support to Rietveld.
http://code.google.com/p/pubsubhubbub
http://code.google.com/p/rietveld/issues/detail?id=155

The server side of the protocol is trivial:
  1. add a &amp;lt;link rel=&amp;quot;hub&amp;quot; href=&amp;quot;hub-server&amp;quot;&amp;gt; tag to all
     feeds that will be pubsubhubbubbed.
  2. every time one of those feeds changes, tell the hub
     with a simple POST request.

I have tested this by adding debug prints to a local hub
server and checking that the server got the right publish
requests.

I can&amp;#39;t quite get the server to work, but I think the bug
is not in my code.  I think that the server expects to be
able to grab the feed and see the feed&amp;#39;s actual URL in
the link rel=&amp;quot;self&amp;quot;, but the default value for that drops
the :port from the URL, and I cannot for the life of me
figure out how to get the Atom generator deep inside
django not to do that, or even where it is doing that,
or even what code is running to generate the Atom feed.
(I thought I knew but I added some assert False statements
and it kept running!)

Ignoring that particular problem, I would appreciate
feedback on the right way to get the two values at
the top of feeds.py marked NOTE(rsc).


</summary></entry><entry><title>rietveld: correct tab handling
</title><link href="http://codereview.appspot.com/124106" rel="alternate"></link><updated>2009-10-03T23:02:17+00:00</updated><author><name>email-address-removed</name></author><id>urn:md5:0a2a4f19bb815101f0ba2904aed7c35a</id><summary type="html">
  This fixes the buggy tab rendering that can be seen at
http://codereview.appspot.com/116075/diff/1/2

The fundamental problem was that the tab code was
not being told what column the text began in, so it
didn&amp;#39;t know where to put the tab stops.  Another problem
was that some of the code assumed that string byte
offsets were the same as column offsets, which is only
true if there are no tabs.

In the process of fixing this, I cleaned up the arguments
to Fold and ExpandTabs and renamed them Break and
_ExpandTabs so that I could be sure that I found all the
call sites.  I also wanted to verify that ExpandTabs was
not being used from outside intra_region_diff.py.


</summary></entry></feed> 	   `

type Feed struct {
	XMLName Name      `xml:"http://www.w3.org/2005/Atom feed"`
	Title   string    `xml:"title"`
	Id      string    `xml:"id"`
	Link    []Link    `xml:"link"`
	Updated time.Time `xml:"updated,attr"`
	Author  Person    `xml:"author"`
	Entry   []Entry   `xml:"entry"`
}

type Entry struct {
	Title   string    `xml:"title"`
	Id      string    `xml:"id"`
	Link    []Link    `xml:"link"`
	Updated time.Time `xml:"updated"`
	Author  Person    `xml:"author"`
	Summary Text      `xml:"summary"`
}

type Link struct {
	Rel  string `xml:"rel,attr,omitempty"`
	Href string `xml:"href,attr"`
}

type Person struct {
	Name     string `xml:"name"`
	URI      string `xml:"uri"`
	Email    string `xml:"email"`
	InnerXML string `xml:",innerxml"`
}

type Text struct {
	Type string `xml:"type,attr,omitempty"`
	Body string `xml:",chardata"`
}

var atomFeed = Feed{
	XMLName: Name{"http://www.w3.org/2005/Atom", "feed"},
	Title:   "Code Review - My issues",
	Link: []Link{
		{Rel: "alternate", Href: "http://codereview.appspot.com/"},
		{Rel: "self", Href: "http://codereview.appspot.com/rss/mine/rsc"},
	},
	Id:      "http://codereview.appspot.com/",
	Updated: ParseTime("2009-10-04T01:35:58+00:00"),
	Author: Person{
		Name:     "rietveld<>",
		InnerXML: "<name>rietveld&lt;&gt;</name>",
	},
	Entry: []Entry{
		{
			Title: "rietveld: an attempt at pubsubhubbub\n",
			Link: []Link{
				{Rel: "alternate", Href: "http://codereview.appspot.com/126085"},
			},
			Updated: ParseTime("2009-10-04T01:35:58+00:00"),
			Author: Person{
				Name:     "email-address-removed",
				InnerXML: "<name>email-address-removed</name>",
			},
			Id: "urn:md5:134d9179c41f806be79b3a5f7877d19a",
			Summary: Text{
				Type: "html",
				Body: `
  An attempt at adding pubsubhubbub support to Rietveld.
http://code.google.com/p/pubsubhubbub
http://code.google.com/p/rietveld/issues/detail?id=155

The server side of the protocol is trivial:
  1. add a &lt;link rel=&quot;hub&quot; href=&quot;hub-server&quot;&gt; tag to all
     feeds that will be pubsubhubbubbed.
  2. every time one of those feeds changes, tell the hub
     with a simple POST request.

I have tested this by adding debug prints to a local hub
server and checking that the server got the right publish
requests.

I can&#39;t quite get the server to work, but I think the bug
is not in my code.  I think that the server expects to be
able to grab the feed and see the feed&#39;s actual URL in
the link rel=&quot;self&quot;, but the default value for that drops
the :port from the URL, and I cannot for the life of me
figure out how to get the Atom generator deep inside
django not to do that, or even where it is doing that,
or even what code is running to generate the Atom feed.
(I thought I knew but I added some assert False statements
and it kept running!)

Ignoring that particular problem, I would appreciate
feedback on the right way to get the two values at
the top of feeds.py marked NOTE(rsc).


`,
			},
		},
		{
			Title: "rietveld: correct tab handling\n",
			Link: []Link{
				{Rel: "alternate", Href: "http://codereview.appspot.com/124106"},
			},
			Updated: ParseTime("2009-10-03T23:02:17+00:00"),
			Author: Person{
				Name:     "email-address-removed",
				InnerXML: "<name>email-address-removed</name>",
			},
			Id: "urn:md5:0a2a4f19bb815101f0ba2904aed7c35a",
			Summary: Text{
				Type: "html",
				Body: `
  This fixes the buggy tab rendering that can be seen at
http://codereview.appspot.com/116075/diff/1/2

The fundamental problem was that the tab code was
not being told what column the text began in, so it
didn&#39;t know where to put the tab stops.  Another problem
was that some of the code assumed that string byte
offsets were the same as column offsets, which is only
true if there are no tabs.

In the process of fixing this, I cleaned up the arguments
to Fold and ExpandTabs and renamed them Break and
_ExpandTabs so that I could be sure that I found all the
call sites.  I also wanted to verify that ExpandTabs was
not being used from outside intra_region_diff.py.


`,
			},
		},
	},
}

const pathTestString = `
<Result>
    <Before>1</Before>
    <Items>
        <Item1>
            <Value>A</Value>
        </Item1>
        <Item2>
            <Value>B</Value>
        </Item2>
        <Item1>
            <Value>C</Value>
            <Value>D</Value>
        </Item1>
        <_>
            <Value>E</Value>
        </_>
    </Items>
    <After>2</After>
</Result>
`

type PathTestItem struct {
	Value string
}

type PathTestA struct {
	Items         []PathTestItem `xml:">Item1"`
	Before, After string
}

type PathTestB struct {
	Other         []PathTestItem `xml:"Items>Item1"`
	Before, After string
}

type PathTestC struct {
	Values1       []string `xml:"Items>Item1>Value"`
	Values2       []string `xml:"Items>Item2>Value"`
	Before, After string
}

type PathTestSet struct {
	Item1 []PathTestItem
}

type PathTestD struct {
	Other         PathTestSet `xml:"Items"`
	Before, After string
}

type PathTestE struct {
	Underline     string `xml:"Items>_>Value"`
	Before, After string
}

var pathTests = []interface{}{
	&PathTestA{Items: []PathTestItem{{"A"}, {"D"}}, Before: "1", After: "2"},
	&PathTestB{Other: []PathTestItem{{"A"}, {"D"}}, Before: "1", After: "2"},
	&PathTestC{Values1: []string{"A", "C", "D"}, Values2: []string{"B"}, Before: "1", After: "2"},
	&PathTestD{Other: PathTestSet{Item1: []PathTestItem{{"A"}, {"D"}}}, Before: "1", After: "2"},
	&PathTestE{Underline: "E", Before: "1", After: "2"},
}

func TestUnmarshalPaths(t *testing.T) {
	for _, pt := range pathTests {
		v := reflect.New(reflect.TypeOf(pt).Elem()).Interface()
		if err := Unmarshal([]byte(pathTestString), v); err != nil {
			t.Fatalf("Unmarshal: %s", err)
		}
		if !reflect.DeepEqual(v, pt) {
			t.Fatalf("have %#v\nwant %#v", v, pt)
		}
	}
}

type BadPathTestA struct {
	First  string `xml:"items>item1"`
	Other  string `xml:"items>item2"`
	Second string `xml:"items"`
}

type BadPathTestB struct {
	Other  string `xml:"items>item2>value"`
	First  string `xml:"items>item1"`
	Second string `xml:"items>item1>value"`
}

type BadPathTestC struct {
	First  string
	Second string `xml:"First"`
}

type BadPathTestD struct {
	BadPathEmbeddedA
	BadPathEmbeddedB
}

type BadPathEmbeddedA struct {
	First string
}

type BadPathEmbeddedB struct {
	Second string `xml:"First"`
}

var badPathTests = []struct {
	v, e interface{}
}{
	{&BadPathTestA{}, &TagPathError{reflect.TypeOf(BadPathTestA{}), "First", "items>item1", "Second", "items"}},
	{&BadPathTestB{}, &TagPathError{reflect.TypeOf(BadPathTestB{}), "First", "items>item1", "Second", "items>item1>value"}},
	{&BadPathTestC{}, &TagPathError{reflect.TypeOf(BadPathTestC{}), "First", "", "Second", "First"}},
	{&BadPathTestD{}, &TagPathError{reflect.TypeOf(BadPathTestD{}), "First", "", "Second", "First"}},
}

func TestUnmarshalBadPaths(t *testing.T) {
	for _, tt := range badPathTests {
		err := Unmarshal([]byte(pathTestString), tt.v)
		if !reflect.DeepEqual(err, tt.e) {
			t.Fatalf("Unmarshal with %#v didn't fail properly:\nhave %#v,\nwant %#v", tt.v, err, tt.e)
		}
	}
}

const OK = "OK"
const withoutNameTypeData = `
<?xml version="1.0" charset="utf-8"?>
<Test3 Attr="OK" />`

type TestThree struct {
	XMLName Name   `xml:"Test3"`
	Attr    string `xml:",attr"`
}

func TestUnmarshalWithoutNameType(t *testing.T) {
	var x TestThree
	if err := Unmarshal([]byte(withoutNameTypeData), &x); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if x.Attr != OK {
		t.Fatalf("have %v\nwant %v", x.Attr, OK)
	}
}

func TestUnmarshalAttr(t *testing.T) {
	type ParamVal struct {
		Int int `xml:"int,attr"`
	}

	type ParamPtr struct {
		Int *int `xml:"int,attr"`
	}

	type ParamStringPtr struct {
		Int *string `xml:"int,attr"`
	}

	x := []byte(`<Param int="1" />`)

	p1 := &ParamPtr{}
	if err := Unmarshal(x, p1); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if p1.Int == nil {
		t.Fatalf("Unmarshal failed in to *int field")
	} else if *p1.Int != 1 {
		t.Fatalf("Unmarshal with %s failed:\nhave %#v,\n want %#v", x, p1.Int, 1)
	}

	p2 := &ParamVal{}
	if err := Unmarshal(x, p2); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if p2.Int != 1 {
		t.Fatalf("Unmarshal with %s failed:\nhave %#v,\n want %#v", x, p2.Int, 1)
	}

	p3 := &ParamStringPtr{}
	if err := Unmarshal(x, p3); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if p3.Int == nil {
		t.Fatalf("Unmarshal failed in to *string field")
	} else if *p3.Int != "1" {
		t.Fatalf("Unmarshal with %s failed:\nhave %#v,\n want %#v", x, p3.Int, 1)
	}
}

type Tables struct {
	HTable string `xml:"http://www.w3.org/TR/html4/ table"`
	FTable string `xml:"http://www.w3schools.com/furniture table"`
}

var tables = []struct {
	xml string
	tab Tables
	ns  string
}{
	{
		xml: `<Tables>` +
			`<table xmlns="http://www.w3.org/TR/html4/">hello</table>` +
			`<table xmlns="http://www.w3schools.com/furniture">world</table>` +
			`</Tables>`,
		tab: Tables{"hello", "world"},
	},
	{
		xml: `<Tables>` +
			`<table xmlns="http://www.w3schools.com/furniture">world</table>` +
			`<table xmlns="http://www.w3.org/TR/html4/">hello</table>` +
			`</Tables>`,
		tab: Tables{"hello", "world"},
	},
	{
		xml: `<Tables xmlns:f="http://www.w3schools.com/furniture" xmlns:h="http://www.w3.org/TR/html4/">` +
			`<f:table>world</f:table>` +
			`<h:table>hello</h:table>` +
			`</Tables>`,
		tab: Tables{"hello", "world"},
	},
	{
		xml: `<Tables>` +
			`<table>bogus</table>` +
			`</Tables>`,
		tab: Tables{},
	},
	{
		xml: `<Tables>` +
			`<table>only</table>` +
			`</Tables>`,
		tab: Tables{HTable: "only"},
		ns:  "http://www.w3.org/TR/html4/",
	},
	{
		xml: `<Tables>` +
			`<table>only</table>` +
			`</Tables>`,
		tab: Tables{FTable: "only"},
		ns:  "http://www.w3schools.com/furniture",
	},
	{
		xml: `<Tables>` +
			`<table>only</table>` +
			`</Tables>`,
		tab: Tables{},
		ns:  "something else entirely",
	},
}

func TestUnmarshalNS(t *testing.T) {
	for i, tt := range tables {
		var dst Tables
		var err error
		if tt.ns != "" {
			d := NewDecoder(strings.NewReader(tt.xml))
			d.DefaultSpace = tt.ns
			err = d.Decode(&dst)
		} else {
			err = Unmarshal([]byte(tt.xml), &dst)
		}
		if err != nil {
			t.Errorf("#%d: Unmarshal: %v", i, err)
			continue
		}
		want := tt.tab
		if dst != want {
			t.Errorf("#%d: dst=%+v, want %+v", i, dst, want)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	// From issue 7535
	const s = `<ex:element xmlns:ex="http://example.com/schema"></ex:element>`
	in := bytes.NewBufferString(s)
	for i := 0; i < 10; i++ {
		out := &bytes.Buffer{}
		d := NewDecoder(in)
		e := NewEncoder(out)

		for {
			t, err := d.Token()
			if err == io.EOF {
				break
			}
			if err != nil {
				fmt.Println("failed:", err)
				return
			}
			e.EncodeToken(t)
		}
		e.Flush()
		in = out
	}
	if got := in.String(); got != s {
		t.Errorf("have: %q\nwant: %q\n", got, s)
	}
}

func TestMarshalNS(t *testing.T) {
	dst := Tables{"hello", "world"}
	data, err := Marshal(&dst)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want := `<Tables><table xmlns="http://www.w3.org/TR/html4/">hello</table><table xmlns="http://www.w3schools.com/furniture">world</table></Tables>`
	str := string(data)
	if str != want {
		t.Errorf("have: %q\nwant: %q\n", str, want)
	}
}

type TableAttrs struct {
	TAttr TAttr
}

type TAttr struct {
	HTable string `xml:"http://www.w3.org/TR/html4/ table,attr"`
	FTable string `xml:"http://www.w3schools.com/furniture table,attr"`
	Lang   string `xml:"http://www.w3.org/XML/1998/namespace lang,attr,omitempty"`
	Other1 string `xml:"http://golang.org/xml/ other,attr,omitempty"`
	Other2 string `xml:"http://golang.org/xmlfoo/ other,attr,omitempty"`
	Other3 string `xml:"http://golang.org/json/ other,attr,omitempty"`
	Other4 string `xml:"http://golang.org/2/json/ other,attr,omitempty"`
}

var tableAttrs = []struct {
	xml string
	tab TableAttrs
	ns  string
}{
	{
		xml: `<TableAttrs xmlns:f="http://www.w3schools.com/furniture" xmlns:h="http://www.w3.org/TR/html4/"><TAttr ` +
			`h:table="hello" f:table="world" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{TAttr{HTable: "hello", FTable: "world"}},
	},
	{
		xml: `<TableAttrs><TAttr xmlns:f="http://www.w3schools.com/furniture" xmlns:h="http://www.w3.org/TR/html4/" ` +
			`h:table="hello" f:table="world" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{TAttr{HTable: "hello", FTable: "world"}},
	},
	{
		xml: `<TableAttrs><TAttr ` +
			`h:table="hello" f:table="world" xmlns:f="http://www.w3schools.com/furniture" xmlns:h="http://www.w3.org/TR/html4/" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{TAttr{HTable: "hello", FTable: "world"}},
	},
	{
		// Default space does not apply to attribute names.
		xml: `<TableAttrs xmlns="http://www.w3schools.com/furniture" xmlns:h="http://www.w3.org/TR/html4/"><TAttr ` +
			`h:table="hello" table="world" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{TAttr{HTable: "hello", FTable: ""}},
	},
	{
		// Default space does not apply to attribute names.
		xml: `<TableAttrs xmlns:f="http://www.w3schools.com/furniture"><TAttr xmlns="http://www.w3.org/TR/html4/" ` +
			`table="hello" f:table="world" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{TAttr{HTable: "", FTable: "world"}},
	},
	{
		xml: `<TableAttrs><TAttr ` +
			`table="bogus" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{},
	},
	{
		// Default space does not apply to attribute names.
		xml: `<TableAttrs xmlns:h="http://www.w3.org/TR/html4/"><TAttr ` +
			`h:table="hello" table="world" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{TAttr{HTable: "hello", FTable: ""}},
		ns:  "http://www.w3schools.com/furniture",
	},
	{
		// Default space does not apply to attribute names.
		xml: `<TableAttrs xmlns:f="http://www.w3schools.com/furniture"><TAttr ` +
			`table="hello" f:table="world" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{TAttr{HTable: "", FTable: "world"}},
		ns:  "http://www.w3.org/TR/html4/",
	},
	{
		xml: `<TableAttrs><TAttr ` +
			`table="bogus" ` +
			`/></TableAttrs>`,
		tab: TableAttrs{},
		ns:  "something else entirely",
	},
}

func TestUnmarshalNSAttr(t *testing.T) {
	for i, tt := range tableAttrs {
		var dst TableAttrs
		var err error
		if tt.ns != "" {
			d := NewDecoder(strings.NewReader(tt.xml))
			d.DefaultSpace = tt.ns
			err = d.Decode(&dst)
		} else {
			err = Unmarshal([]byte(tt.xml), &dst)
		}
		if err != nil {
			t.Errorf("#%d: Unmarshal: %v", i, err)
			continue
		}
		want := tt.tab
		if dst != want {
			t.Errorf("#%d: dst=%+v, want %+v", i, dst, want)
		}
	}
}

func TestMarshalNSAttr(t *testing.T) {
	src := TableAttrs{TAttr{"hello", "world", "en_US", "other1", "other2", "other3", "other4"}}
	data, err := Marshal(&src)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want := `<TableAttrs><TAttr xmlns:json_1="http://golang.org/2/json/" xmlns:json="http://golang.org/json/" xmlns:_xmlfoo="http://golang.org/xmlfoo/" xmlns:_xml="http://golang.org/xml/" xmlns:furniture="http://www.w3schools.com/furniture" xmlns:html4="http://www.w3.org/TR/html4/" html4:table="hello" furniture:table="world" xml:lang="en_US" _xml:other="other1" _xmlfoo:other="other2" json:other="other3" json_1:other="other4"></TAttr></TableAttrs>`
	str := string(data)
	if str != want {
		t.Errorf("Marshal:\nhave: %#q\nwant: %#q\n", str, want)
	}

	var dst TableAttrs
	if err := Unmarshal(data, &dst); err != nil {
		t.Errorf("Unmarshal: %v", err)
	}

	if dst != src {
		t.Errorf("Unmarshal = %q, want %q", dst, src)
	}
}

type MyCharData struct {
	body string
}

func (m *MyCharData) UnmarshalXML(d *Decoder, start StartElement) error {
	for {
		t, err := d.Token()
		if err == io.EOF { // found end of element
			break
		}
		if err != nil {
			return err
		}
		if char, ok := t.(CharData); ok {
			m.body += string(char)
		}
	}
	return nil
}

var _ Unmarshaler = (*MyCharData)(nil)

func (m *MyCharData) UnmarshalXMLAttr(attr Attr) error {
	panic("must not call")
}

type MyAttr struct {
	attr string
}

func (m *MyAttr) UnmarshalXMLAttr(attr Attr) error {
	m.attr = attr.Value
	return nil
}

var _ UnmarshalerAttr = (*MyAttr)(nil)

type MyStruct struct {
	Data *MyCharData
	Attr *MyAttr `xml:",attr"`

	Data2 MyCharData
	Attr2 MyAttr `xml:",attr"`
}

func TestUnmarshaler(t *testing.T) {
	xml := `<?xml version="1.0" encoding="utf-8"?>
		<MyStruct Attr="attr1" Attr2="attr2">
		<Data>hello <!-- comment -->world</Data>
		<Data2>howdy <!-- comment -->world</Data2>
		</MyStruct>
	`

	var m MyStruct
	if err := Unmarshal([]byte(xml), &m); err != nil {
		t.Fatal(err)
	}

	if m.Data == nil || m.Attr == nil || m.Data.body != "hello world" || m.Attr.attr != "attr1" || m.Data2.body != "howdy world" || m.Attr2.attr != "attr2" {
		t.Errorf("m=%#+v\n", m)
	}
}

type Pea struct {
	Cotelydon string
}

type Pod struct {
	Pea interface{} `xml:"Pea"`
}

// https://golang.org/issue/6836
func TestUnmarshalIntoInterface(t *testing.T) {
	pod := new(Pod)
	pod.Pea = new(Pea)
	xml := `<Pod><Pea><Cotelydon>Green stuff</Cotelydon></Pea></Pod>`
	err := Unmarshal([]byte(xml), pod)
	if err != nil {
		t.Fatalf("failed to unmarshal %q: %v", xml, err)
	}
	pea, ok := pod.Pea.(*Pea)
	if !ok {
		t.Fatalf("unmarshalled into wrong type: have %T want *Pea", pod.Pea)
	}
	have, want := pea.Cotelydon, "Green stuff"
	if have != want {
		t.Errorf("failed to unmarshal into interface, have %q want %q", have, want)
	}
}
