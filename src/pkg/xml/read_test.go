// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"reflect";
	"testing";
)

// Stripped down Atom feed data structures.

func TestUnmarshalFeed(t *testing.T) {
	var f Feed;
	if err := Unmarshal(StringReader(rssFeedString), &f); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if !reflect.DeepEqual(f, rssFeed) {
		t.Fatalf("have %#v\nwant %#v\n\n%#v", f)
	}
}

// hget http://codereview.appspot.com/rss/mine/rsc
const rssFeedString = `
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en-us"><title>Code Review - My issues</title><link href="http://codereview.appspot.com/" rel="alternate"></link><link href="http://codereview.appspot.com/rss/mine/rsc" rel="self"></link><id>http://codereview.appspot.com/</id><updated>2009-10-04T01:35:58+00:00</updated><author><name>rietveld</name></author><entry><title>rietveld: an attempt at pubsubhubbub
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


</summary></entry></feed>`

type Feed struct {
	XMLName	Name	"http://www.w3.org/2005/Atom feed";
	Title	string;
	Id	string;
	Link	[]Link;
	Updated	Time;
	Author	Person;
	Entry	[]Entry;
}

type Entry struct {
	Title	string;
	Id	string;
	Link	[]Link;
	Updated	Time;
	Author	Person;
	Summary	Text;
}

type Link struct {
	Rel	string	"attr";
	Href	string	"attr";
}

type Person struct {
	Name	string;
	URI	string;
	Email	string;
}

type Text struct {
	Type	string	"attr";
	Body	string	"chardata";
}

type Time string

var rssFeed = Feed{
	XMLName: Name{"http://www.w3.org/2005/Atom", "feed"},
	Title: "Code Review - My issues",
	Link: []Link{
		Link{Rel: "alternate", Href: "http://codereview.appspot.com/"},
		Link{Rel: "self", Href: "http://codereview.appspot.com/rss/mine/rsc"},
	},
	Id: "http://codereview.appspot.com/",
	Updated: "2009-10-04T01:35:58+00:00",
	Author: Person{
		Name: "rietveld",
	},
	Entry: []Entry{
		Entry{
			Title: "rietveld: an attempt at pubsubhubbub\n",
			Link: []Link{
				Link{Rel: "alternate", Href: "http://codereview.appspot.com/126085"},
			},
			Updated: "2009-10-04T01:35:58+00:00",
			Author: Person{
				Name: "email-address-removed",
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
		Entry{
			Title: "rietveld: correct tab handling\n",
			Link: []Link{
				Link{Rel: "alternate", Href: "http://codereview.appspot.com/124106"},
			},
			Updated: "2009-10-03T23:02:17+00:00",
			Author: Person{
				Name: "email-address-removed",
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
