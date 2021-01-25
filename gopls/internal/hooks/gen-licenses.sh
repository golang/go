#!/bin/bash -eu

# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -o pipefail

output=$1
tempfile=$(mktemp)
cd $(dirname $0)

cat > $tempfile <<END
// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate ./gen-licenses.sh licenses.go
package hooks

const licensesText = \`
END

# List all the modules gopls depends on, except other golang.org modules, which
# are known to have the same license.
mods=$(go list -deps -f '{{with .Module}}{{.Path}}{{end}}' golang.org/x/tools/gopls | sort -u | grep -v golang.org)
for mod in $mods; do
  # Find the license file, either LICENSE or COPYING, and add it to the result.
  dir=$(go list -m -f {{.Dir}} $mod)
  license=$(ls -1 $dir | egrep -i '^(LICENSE|COPYING)$')
  echo "-- $mod $license --" >> $tempfile
  echo >> $tempfile
  sed 's/^-- / &/' $dir/$license >> $tempfile
  echo >> $tempfile
done

echo "\`" >> $tempfile
mv $tempfile $output