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

# Check that the go command exists
if ! go help >/dev/null 2>&1; then
	echo "The go command is not in your PATH." >&2
	exit 2
fi

eval $(go env)
if ! [ -x $GOTOOLDIR/cov -a -x $GOTOOLDIR/prof ]; then
	echo "You don't need to run sudo.bash." >&2
	exit 2
fi

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
	sudo cp $GOTOOLDIR/$i /usr/local/bin/go$i
	sudo chgrp procmod /usr/local/bin/go$i
	sudo chmod g+s /usr/local/bin/go$i
done
