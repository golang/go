// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


extern	char	gotypestrings[];	// 4-byte count followed by byte[count]

void FLUSH(void*);

typedef	struct	String	String;
struct	String
{
	char*	str;
	char	len[4];
	char	cap[4];
};

void
reflect·typestrings(String str)
{
	char *s;
	int i;

	s = gotypestrings;

	// repeat the count twice
	// once for len, once for cap
	for(i=0; i<4; i++) {
		str.len[i] = s[i];
		str.cap[i] = s[i];
	}

	// and the pointer
	str.str = s+4;

	FLUSH(&str);
}
