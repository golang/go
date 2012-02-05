#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

case "`uname`" in
Darwin)
	;;
*)
	exit 0
esac

if [[ ! -d /usr/local/bin ]]; then
	echo 1>&2 'sudo.bash: problem with /usr/local/bin; cannot install tools.'
	exit 2
fi

cd $(dirname $0)
for i in prof cov
do
	# Remove old binaries if present
	sudo rm -f /usr/local/bin/6$i
	# Install new binaries
	sudo cp ../bin/tool/$i /usr/local/bin/go$i
	sudo chgrp procmod /usr/local/bin/go$i
	sudo chmod g+s /usr/local/bin/go$i
done
