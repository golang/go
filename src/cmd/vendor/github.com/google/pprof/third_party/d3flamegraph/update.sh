#  Copyright 2021 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#!/usr/bin/env bash

set -eu
set -o pipefail

D3FLAMEGRAPH_CSS="d3-flamegraph.css"

cd $(dirname $0)

generate_d3_flame_graph_go() {
    npm install
    # https://stackoverflow.com/a/21199041/171898
    local d3_js=$(cat d3.js | sed 's/`/`+"`"+`/g')
    local d3_css=$(cat "node_modules/d3-flame-graph/dist/${D3FLAMEGRAPH_CSS}")

    cat <<-EOF > d3_flame_graph.go
// D3.js is a JavaScript library for manipulating documents based on data.
// https://github.com/d3/d3
// See D3_LICENSE file for license details

// d3-flame-graph is a D3.js plugin that produces flame graphs from hierarchical data.
// https://github.com/spiermar/d3-flame-graph
// See D3_FLAME_GRAPH_LICENSE file for license details

package d3flamegraph

// JSSource returns the d3 and d3-flame-graph JavaScript bundle
const JSSource = \`

$d3_js
\`

// CSSSource returns the $D3FLAMEGRAPH_CSS file
const CSSSource = \`
$d3_css
\`

EOF
    gofmt -w d3_flame_graph.go
}

get_licenses() {
    cp node_modules/d3-selection/LICENSE D3_LICENSE
    cp node_modules/d3-flame-graph/LICENSE D3_FLAME_GRAPH_LICENSE
}

get_licenses
generate_d3_flame_graph_go
