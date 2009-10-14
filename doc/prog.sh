#!/bin/sh
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# generate HTML for a program excerpt.
# first arg is file name
# second arg is awk pattern to match start line
# third arg is awk pattern to stop processing
#
# missing third arg means print one line
# third arg "END" means proces rest of file
# missing second arg means process whole file
#
# examples:
#
#	prog.sh foo.go                       # whole file
#	prog.sh foo.go "/^func.main/"        # signature of main
#	prog.sh foo.go "/^func.main/" "/^}/  # body of main
#
# non-blank lines are annotated with line number in file

# line numbers are printed %.2d to make them equal-width for nice formatting.
# the format gives a leading 0.  the format %2d gives a leading space but
# that appears to confuse sanjay's makehtml formatter into bungling quotes
# because it makes some lines look indented.

echo "<pre> <!-- $* -->"

case $# in
3)
	if test "$3" = "END"  # $2 to end of file
	then
		awk '
			function LINE() { printf("%.2d\t%s\n", NR, $0) }
			BEGIN { printing = 0 }
			'$2' { printing = 1; LINE(); getline }
			printing { if($0 ~ /./) { LINE() } else { print "" } }
		'
	else	# $2 through $3
		awk '
			function LINE() { printf("%.2d\t%s\n", NR, $0) }
			BEGIN { printing = 0 }
			'$2' { printing = 1; LINE(); getline }
			'$3' && printing { if(printing) {printing = 0; LINE(); exit} }
			printing { if($0 ~ /./) { LINE() } else { print "" } }
		'
	fi
	;;
2)	# one line
	awk '
		function LINE() { printf("%.2d\t%s\n", NR, $0) }
		'$2' { LINE(); getline; exit }
	'
	;;
1)	# whole file
	awk '
		function LINE() { printf("%.2d\t%s\n", NR, $0) }
		{ if($0 ~ /./) { LINE() } else { print "" } }
	'
	;;
*)
	echo >&2 usage: prog.sh file.go /func.main/ /^}/
esac <$1 |
sed '
	s/&/\&amp;/g
	s/"/\&quot;/g
	s/</\&lt;/g
	s/>/\&gt;/g
'

echo '</pre>'
