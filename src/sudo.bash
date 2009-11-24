#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

case "`uname`" in
Darwin)
	;;
*)
	exit 0
esac

for i in prof cov
do
	sudo cp "$GOROOT"/src/cmd/$i/6$i /usr/local/bin/6$i
	sudo chgrp procmod /usr/local/bin/6$i
	sudo chmod g+s /usr/local/bin/6$i
done
