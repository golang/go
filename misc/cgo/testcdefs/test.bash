# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Just add issue file prefixes to this list if more issues come up
FILE_PREFIXES="cdefstest"

for FP in $FILE_PREFIXES
do 
  go tool cgo -cdefs ${FP}.go > ${FP}.h
done

go build . && ./testcdefs
EXIT=$?
rm -rf _obj testcdefs *.h
exit $EXIT
