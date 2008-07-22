# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

6g sys.go
echo '1,/((/d
/))/+1,$d
1,$s/foop/sys/g
1,$s/^[ 	]*/	"/g
1,$s/$/\\n"/g
1i
char*	sysimport =
.
$a
;

.
w sysimport.c
q' | /bin/ed sys.6
