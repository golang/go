// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define NOPLAN9DEFINES
#include <u.h>
#include <libc.h>

char*
p9ctime(long t)
{
	static char buf[100];
	time_t tt;
	struct tm *tm;
	
	tt = t;
	tm = localtime(&tt);
	snprint(buf, sizeof buf, "%3.3s %3.3s %02d %02d:%02d:%02d %3.3s %d\n",
		&"SunMonTueWedThuFriSat"[tm->tm_wday*3],
		&"JanFebMarAprMayJunJulAugSepOctNovDec"[tm->tm_mon*3],
		tm->tm_mday,
		tm->tm_hour,
		tm->tm_min,
		tm->tm_sec,
		"XXX",  // tm_zone is unavailable on windows, and no one cares
		tm->tm_year + 1900);
	return buf;
}
