#!/usr/bin/env bash

# Copyright 2018 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Regression tests for golang.org.
# Usage: ./regtest.bash https://golang.org/

#TODO: turn this into a Go program. maybe behind a build tag and "go run regtest.go <url>"

set -e

addr="$(echo $1 | sed -e 's/\/$//')"
if [ -z "$addr" ]; then
	echo "usage: $0 <addr>" 1>&2
	echo "example: $0 https://20180928t023837-dot-golang-org.appspot.com/" 1>&2
	exit 1
fi

set -u

# fetch url, check the response with a regexp.
fetch() {
	curl -s "${addr}$1" | grep "$2" > /dev/null
}
fatal() {
	log "$1"
	exit 1
}
log() {
	echo "$1" 1>&2
}
logn() {
	echo -n "$1" 1>&2
}

log "Checking FAQ..."
fetch /doc/faq 'What is the purpose of the project' || {
	fatal "FAQ did not match."
}

log "Checking package listing..."
fetch /pkg/ 'Package tar' || {
	fatal "package listing page did not match."
}

log "Checking os package..."
fetch /pkg/os/ 'func Open' || {
	fatal "os package page did not match."
}

log "Checking robots.txt..."
fetch /robots.txt 'Disallow: /search' || {
	fatal "robots.txt did not match."
}

log "Checking /change/ redirect..."
fetch /change/75944e2e3a63 'bdb10cf' || {
	fatal "/change/ direct did not match."
}

log "Checking /dl/ page has data..."
fetch /dl/ 'go1.11.windows-amd64.msi' || {
	fatal "/dl/ did not match."
}

log "Checking /dl/?mode=json page has data..."
fetch /dl/?mode=json 'go1.11.windows-amd64.msi' || {
	fatal "/dl/?mode=json did not match."
}

log "Checking shortlinks (/s/go2design)..."
fetch /s/go2design 'proposal.*Found' || {
	fatal "/s/go2design did not match."
}

log "Checking analytics on pages..."
ga_id="UA-11222381-2"
fetch / $ga_id || fatal "/ missing GA."
fetch /dl/ $ga_id || fatal "/dl/ missing GA."
fetch /project/ $ga_id || fatal "/project missing GA."
fetch /pkg/context/ $ga_id || fatal "/pkg/context missing GA."

log "Checking search..."
fetch /search?q=IsDir 'src/os/types.go' || {
	fatal "search result did not match."
}

log "Checking compile service..."
compile="curl -s ${addr}/compile"

p="package main; func main() { print(6*7); }"
$compile --data-urlencode "body=$p" | tee /tmp/compile.out | grep '^{"compile_errors":"","output":"42"}$' > /dev/null || {
	cat /tmp/compile.out
	fatal "compile service output did not match."
}

$compile --data-urlencode "body=//empty" | tee /tmp/compile.out | grep "expected 'package', found 'EOF'" > /dev/null || {
	cat /tmp/compile.out
	fatal "compile service error output did not match."
}

# Check API version 2
d="version=2&body=package+main%3Bimport+(%22fmt%22%3B%22time%22)%3Bfunc+main()%7Bfmt.Print(%22A%22)%3Btime.Sleep(time.Second)%3Bfmt.Print(%22B%22)%7D"
$compile --data "$d" | grep '^{"Errors":"","Events":\[{"Message":"A","Kind":"stdout","Delay":0},{"Message":"B","Kind":"stdout","Delay":1000000000}\]}$' > /dev/null || {
	fatal "compile service v2 output did not match."
}

log "All OK"
