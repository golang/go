#!/bin/bash
# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Test the code review plugin.
# Assumes a local Rietveld is running using the App Engine SDK
# at http://localhost:7777/
#
# dev_appserver.py --port 7777 $HOME/pub/rietveld

codereview_script=$(pwd)/codereview.py
server=localhost:7777
master=/tmp/go.test
clone1=/tmp/go.test1
clone2=/tmp/go.test2
export HGEDITOR=true

must() {
	if ! "$@"; then
		echo "$@" failed >&2
		exit 1
	fi
}

not() {
	if "$@"; then
		false
	else
		true
	fi
}

status() {
	echo '+++' "$@" >&2
}

firstcl() {
	hg pending | sed 1q | tr -d ':'
}

# Initial setup.
status Create repositories.
rm -rf $master $clone1 $clone2
mkdir $master
cd $master
must hg init .
echo Initial state >file
must hg add file
must hg ci -m 'first commit' file
must hg clone $master $clone1
must hg clone $master $clone2

echo "
[ui]
username=Grace R Emlin <gre@golang.org>
[extensions]
codereview=$codereview_script
[codereview]
testing=true
server=$server
" >>$clone1/.hg/hgrc
cp $clone1/.hg/hgrc $clone2/.hg/hgrc

status Codereview should be disabled.
cd $clone1
must hg status
must not hg pending

status Enabling code review.
must mkdir lib lib/codereview
must touch lib/codereview/codereview.cfg

status Code review should work even without CONTRIBUTORS.
must hg pending

status Add CONTRIBUTORS.
echo 'Grace R Emlin <gre@golang.org>' >CONTRIBUTORS
must hg add lib/codereview/codereview.cfg CONTRIBUTORS

status First submit.
must hg submit --tbr gre@golang.org -m codereview \
	lib/codereview/codereview.cfg CONTRIBUTORS

status Should see change in other client.
cd $clone2
must hg pull -u
must test -f lib/codereview/codereview.cfg
must test -f CONTRIBUTORS

test_clpatch() {
	# The email address must be test@example.com to match
	# the test code review server's default user.
	# Clpatch will check.
	
	cd $clone1
	# dev_appserver.py used to crash with UTF-8 input.
	if true; then
		status Using UTF-8.
		name="Grácè T Emlïn <test@example.com>"
	else
		status Using ASCII.
		name="Grace T Emlin <test@example.com>"
	fi
	echo "$name" >>CONTRIBUTORS
	cat .hg/hgrc | sed "s/Grace.*/$name/" >/tmp/x && mv /tmp/x .hg/hgrc
	echo "
Reviewer: gre@golang.org
Description:
	CONTRIBUTORS: add $name
Files:
	CONTRIBUTORS
"	| must hg change -i
	num=$(hg pending | sed 1q | tr -d :)
	
	status Patch CL.
	cd $clone2
	must hg clpatch $num
	must [ "$num" = "$(firstcl)" ]
	must hg submit --tbr gre@golang.org $num
	
	status Issue should be open with no reviewers.
	must curl http://$server/api/$num >/tmp/x
	must not grep '"closed":true' /tmp/x
	must grep '"reviewers":\[\]' /tmp/x
	
	status Sync should close issue.
	cd $clone1
	must hg sync
	must curl http://$server/api/$num >/tmp/x
	must grep '"closed":true' /tmp/x
	must grep '"reviewers":\[\]' /tmp/x
	must [ "$(firstcl)" = "" ]
}

test_reviewer() {
	status Submit without reviewer should fail.
	cd $clone1
	echo dummy >dummy
	must hg add dummy
	echo '
Description:
	no reviewer
Files:
	dummy
'	| must hg change -i
	num=$(firstcl)
	must not hg submit $num
	must hg revert dummy
	must rm dummy
	must hg change -d $num
}

test_linearity() {
	status Linearity of changes.
	cd $clone1
	echo file1 >file1
	must hg add file1
	echo '
Reviewer: gre@golang.org
Description: file1
Files: file1
	' | must hg change -i
	must hg submit --tbr gre@golang.org $(firstcl)
	
	cd $clone2
	echo file2 >file2
	must hg add file2
	echo '
Reviewer: gre@golang.org
Description: file2
Files: file2
	' | must hg change -i
	must not hg submit --tbr gre@golang.org $(firstcl)
	must hg sync
	must hg submit --tbr gre@golang.org $(firstcl)
}

test_restrict() {
	status Cannot use hg ci.
	cd $clone1
	echo file1a >file1a
	hg add file1a
	must not hg ci -m commit file1a
	must rm file1a
	must hg revert file1a
	
	status Cannot use hg rollback.
	must not hg rollback
	
	status Cannot use hg backout
	must not hg backout -r -1
}

test_reviewer
test_clpatch
test_linearity
test_restrict
status ALL TESTS PASSED.
