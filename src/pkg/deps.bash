#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

eval $(gomake --no-print-directory -f ../Make.inc go-env)

OUT="Make.deps"
TMP="Make.deps.tmp"

if [ -f $OUT ] && ! [ -w $OUT ]; then
	echo "$0: $OUT is read-only; aborting." 1>&2
	exit 1
fi

# Get list of directories from Makefile
dirs=$(gomake --no-print-directory echo-dirs)
dirpat=$(echo $dirs C | awk '{
	for(i=1;i<=NF;i++){ 
		x=$i
		gsub("/", "\\/", x)
		printf("/^(%s)$/\n", x)
	}
}')

# Append old names of renamed packages. TODO: clean up after renaming.
dirpat="$dirpat
/^(asn1)$/
/^(big)$/
/^(cmath)$/
/^(csv)$/
/^(exec)$/
/^(exp\/template\/html)$/
/^(gob)$/
/^(http)/
/^(http\/cgi)$/
/^(http\/fcgi)$/
/^(http\/httptest)$/
/^(http\/pprof)$/
/^(json)$/
/^(mail)$/
/^(rand)$/
/^(rpc)$/
/^(scanner)$/
/^(smtp)$/
/^(syslog)$/
/^(tabwriter)$/
/^(url)$/
/^(template)$/
/^(template\/parse)$/
/^(utf16)$/
/^(utf8)$/
/^(xml)$/
"

for dir in $dirs; do (
	cd $dir >/dev/null || exit 1

	sources=$(sed -n 's/^[ 	]*\([^ 	]*\.go\)[ 	]*\\*[ 	]*$/\1/p' Makefile)
	sources=$(echo $sources | sed 's/\$(GOOS)/'$GOOS'/g')
	sources=$(echo $sources | sed 's/\$(GOARCH)/'$GOARCH'/g')
	# /dev/null here means we get an empty dependency list if $sources is empty
	# instead of listing every file in the directory.
	sources=$(ls $sources /dev/null 2> /dev/null)  # remove .s, .c, etc.

	deps=$(
		sed -n '/^import.*"/p; /^import[ \t]*(/,/^)/p' $sources /dev/null |
		cut -d '"' -f2 |
		awk "$dirpat" |
		grep -v "^$dir\$" |
		sed 's/$/.install/' |
		# TODO: rename the dependencies for renamed directories.  TODO: clean up after renaming.
		sed 's;^asn1.install$;encoding/asn1.install;
		s;^big.install$;math/big.install;
		s;^cmath.install$;math/cmplx.install;
		s;^csv.install$;encoding/csv.install;
		s;^exec.install$;os/exec.install;
		s;^exp/template/html.install$;html/template.install;
		s;^gob.install$;encoding/gob.install;
		s;^http.install$;net/http.install;
		s;^http/cgi.install$;net/http/cgi.install;
		s;^http/fcgi.install$;net/http/fcgi.install;
		s;^http/httptest.install$;net/http/httptest.install;
		s;^http/pprof.install$;net/http/pprof.install;
		s;^json.install$;encoding/json.install;
		s;^mail.install$;net/mail.install;
		s;^rpc.install$;net/rpc.install;
		s;^rpc/jsonrpc.install$;net/rpc/jsonrpc.install;
		s;^scanner.install$;text/scanner.install;
		s;^smtp.install$;net/smtp.install;
		s;^syslog.install$;log/syslog.install;
		s;^tabwriter.install$;text/tabwriter.install;
		s;^template.install$;text/template.install;
		s;^template/parse.install$;text/template/parse.install;
		s;^rand.install$;math/rand.install;
		s;^url.install$;net/url.install;
		s;^utf16.install$;unicode/utf16.install;
		s;^utf8.install$;unicode/utf8.install;
		s;^xml.install$;encoding/xml.install;' |
		# TODO: end of renamings.
		sed 's;^C\.install;runtime/cgo.install;' |
		sort -u
	)

	echo $dir.install: $deps
) done > $TMP

mv $TMP $OUT

if (egrep -v '^(exp|old)/' $OUT | egrep -q " (exp|old)/"); then
	echo "$0: $OUT contains dependencies to exp or old packages"
        exit 1
fi
